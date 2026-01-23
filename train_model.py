import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix
import joblib
from flask import Flask,render_template,make_response,redirect,url_for,request
from flask import jsonify
import os
import shutil
import requests
import plotly.graph_objects as go

app=Flask(__name__,template_folder="templates",static_folder="static",static_url_path="/")

#---------------bro dont forget to include the stat report based on the cm and cr graph for how the model improved iteself
class_names=[]
class_count=0
best_cm=[]
best_cr={}
cm=best_cm
cr=best_cr

@app.route("/trainNewModel",methods=["POST"])
def trainNewModel():
    file_path = './data/bank_text_balanced_with_upi.csv'
    #file_path = './data/totData.csv'
    df = pd.read_csv(file_path)
    X = df['narrative'].astype(str)
    y = df['product'].astype(str)
    url="http://127.0.0.1:5001/mlmodel/getMisclassification"
    response=requests.post(url)
    misclassified_data=response.json()["misclassified"]
    if misclassified_data!=[]:
        mis_df=pd.DataFrame(misclassified_data).rename(
            columns={"complaint":"narrative","actual_cls":"product"}
            )[["narrative","product"]]
        df=pd.concat([df,mis_df],ignore_index=True)
    df = df.drop_duplicates()
    
    X = df['narrative'].astype(str)
    y = df['product'].astype(str)

    df.to_csv("./data/bank_text_balanced_with_upi.csv")

    # along with the X and Y add new rows from misclassified_data.csv

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    X_train_reset = X_train.reset_index(drop=True)
    X_test_reset = X_test.reset_index(drop=True)

    print("\nLoading SBERT model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Encoding train...")
    X_train_sbert = csr_matrix(sbert_model.encode(X_train_reset.tolist()))
    print("Encoding test...")
    X_test_sbert = csr_matrix(sbert_model.encode(X_test_reset.tolist()))

    print("\nBuilding TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1,2), min_df=3, max_df=0.9)
    X_train_tfidf = tfidf.fit_transform(X_train_reset)
    X_test_tfidf = tfidf.transform(X_test_reset)

    print("\nCombining Features...")
    X_train_combined = hstack([X_train_tfidf, X_train_sbert])
    X_test_combined = hstack([X_test_tfidf, X_test_sbert])

    # Models
    base_estimators = [
        ('svc', LinearSVC(C=0.3, class_weight='balanced', max_iter=5000)),
        ('logreg', LogisticRegression(C=0.3, max_iter=5000, solver='liblinear', class_weight='balanced', multi_class='ovr'))
    ]
    final_estimator = LogisticRegression(C=0.3, max_iter=5000, solver='lbfgs', class_weight='balanced')

    stacking_clf = StackingClassifier(
        estimators=base_estimators, final_estimator=final_estimator, cv=3, n_jobs=-1)

    print("\nTraining final Stacking model...")
    stacking_clf.fit(X_train_combined, y_train)

    # Accuracy report
    y_pred = stacking_clf.predict(X_test_combined)
    
    n_classes=len(label_encoder.classes_) # say 5
    class_indices=np.arange(n_classes) # [0,1,2,3,4]
    class_names=label_encoder.inverse_transform(class_indices).tolist()

    c_cm=confusion_matrix(y_test,y_pred,labels=class_indices)
    c_cm_list=c_cm.tolist()

    c_cr=classification_report(
        y_test,y_pred,
        labels=class_indices,
        target_names=class_names,
        output_dict=True)


    # Save model + TF-IDF + Label Encoder
    joblib.dump(stacking_clf, "./models/model.pkl")
    joblib.dump(tfidf, "./models/tfidf.pkl")
    joblib.dump(label_encoder, "./models/label_encoder.pkl")

    print("\n🎉 Model Training Complete & Saved Successfully!")

    response=make_response()

    global cm;global cr
    cm=c_cm_list
    cr=c_cr
    print(cm)
    print(cr)

    return redirect(url_for('threshold'))

#model improvement  side --------------------------------------------

@app.route("/threshold")
def threshold():
    global best_cm
    global best_cr
    global cm
    global cr
    return render_template("threshold.html",best_cm=best_cm,best_cr=best_cr,class_names=class_names,cm=cm,cr=cr)

def file_transfer(src_dir,dest_dir):
    for fname in os.listdir(src_dir):
        src=os.path.join(src_dir,fname)
        dest=os.path.join(dest_dir,fname)
        if os.path.isfile(src):
            shutil.copy2(src,dest)   

def delMisses():
    url="http://127.0.0.1:5001/mlmodel/delMisclses"
    response=requests.post(url)
    print(response.json())

@app.route("/promote",methods=["POST"])
def promote():
    src_dir="./models"
    dest_dir="./models/bestModels"
    file_transfer(src_dir,dest_dir)
    src_dir="./data"
    dest_dir="./data/bestDataSet"
    file_transfer(src_dir,dest_dir)

    global best_cm;global best_cr
    global cm;global cr
    best_cm=cm
    best_cr=cr

    versionID=request.form.get("model_version_id")
    notes=request.form.get("notes")

    url="http://127.0.0.1:5001/mlmodel/setConfusionMatrix"
    response=requests.post(url,json={"confusion_matrix":best_cm})
    print(response.json())

    url="http://127.0.0.1:5001/mlmodel/setMetrics"
    response=requests.post(url,json=best_cr)
    print(response.json())

    url="http://127.0.0.1:5001/mlmodel/getMisclsCount"
    response=requests.post(url)
    print(response.json())

    noOfSamples=response.json()['count']
    url="http://127.0.0.1:5001/mlmodel/setModelVersion"
    response=requests.post(url,json={
        "model_version":{
            "version_name":versionID,
            "notes":notes,
            "trained_on_samples":noOfSamples
        }
    })
    print(response.json())

    # set model-level metrics
    response=requests.post(
        "http://127.0.0.1:5001/mlmodel/setModelLevelTrace",
        json={
            "mlt": {
                "model_version_id": versionID,
                "accuracy": best_cr["accuracy"],
                "macro_f1": best_cr["macro avg"]["f1-score"],
                "macro_recall": best_cr["macro avg"]["recall"],
                "macro_support": best_cr["macro avg"]["support"],
                "macro_precision": best_cr["macro avg"]["precision"],
                "weighted_f1": best_cr["weighted avg"]["f1-score"],
                "weighted_recall": best_cr["weighted avg"]["recall"],
                "weighted_support": best_cr["weighted avg"]["support"],
                "weighted_precision": best_cr["weighted avg"]["precision"]
            }
        }
    )
    print(response.json())

    # set class-level metrics (random)--------
    global class_names
    classes = class_names


    class_metrics = []
    for cls in classes:
        class_metrics.append({
            "model_version_id": versionID,
            "class_name": cls,
            "recall": best_cr[cls]["recall"],
            "f1": best_cr[cls]["f1-score"],
            "support": best_cr[cls]["support"]
        })

    response=requests.post(
        "http://127.0.0.1:5001/mlmodel/setClassLevelTrace",
        json={"clt": class_metrics}
    )
    print(response.json())

    # set error-level metrics------------
    global class_count
    matrix = []
    for i in range(class_count):
        for j in range(class_count):
            matrix.append({
                "predicted_class": classes[i],
                "actual_class": classes[j],
                "count": best_cm[i][j]
            })

    response=requests.post(
        "http://127.0.0.1:5001/mlmodel/setErrorLevelTrace",
        json={
            "elt": {
                "model_version_id": versionID,
                "matrix": matrix
            }
        }
    )
    print(response.json())
    print("1")

    # set version summary metrics
    worst_class = ""
    worst_f1 = float('inf')

    for cls in classes:
        if best_cr[cls]['f1-score'] <= worst_f1:
            worst_f1 = best_cr[cls]['f1-score']
            worst_class = cls

    print("2")
    response=requests.post( #set version metrics
        "http://127.0.0.1:5001/mlmodel/setVersionMetrics",
        json={
            "version_summary": {
                "model_version_id": versionID,
                "accuracy": best_cr['accuracy'],
                "worst_class_f1": worst_f1,
                "worst_class_name": worst_class,
                "misclassification_rate": float(np.trace(best_cm))
            }
        }
    )
    print("3")
    print(response.json())
    print("4")
    delMisses()
    return redirect(url_for('threshold'))

@app.route("/rollback",methods=["POST"])
def rollback():
    src_dir="./models/bestModels"
    dest_dir="./models"
    file_transfer(src_dir,dest_dir)
    src_dir="./data/bestDataSet"
    dest_dir="./data"
    file_transfer(src_dir,dest_dir)
    delMisses()
    return redirect(url_for('threshold'))

def init_metrics():
    url="http://127.0.0.1:5001/mlmodel/getClasses"
    response=requests.post(url)
    classes=response.json()['classes']
    global class_names
    class_names=classes

    url="http://127.0.0.1:5001/mlmodel/getClsCount"
    response=requests.post(url)
    count=response.json()['ClassCount']
    global class_count
    class_count=count

    print({"class_names":class_names,"class_count":class_count})

    url="http://127.0.0.1:5001/mlmodel/getConfusionMatrix"
    response=requests.post(url)
    res_cm=response.json()['confusion_matrix']
    global best_cm


    url="http://127.0.0.1:5001/mlmodel/metrics/classes"
    response=requests.post(url)
    crc=response.json()['class_metrics']
    url="http://127.0.0.1:5001/mlmodel/metrics/model"
    response=requests.post(url)
    crm=response.json()['model_metrics']
    start=0
    best_cm=[[0 for _ in range(class_count)] for _ in range(class_count)]
    for i in range(class_count):
        for j in range(class_count):
            best_cm[i][j]=res_cm[start]['ClassificationCount']
            start+=1

    global best_cr
    for row in crc:
        best_cr[row['class_name']]={"f1-score":row['f1_score'],
                                    "precision":row['precision'],
                                    "recall":row['recall'],
                                    "support":row['support']}
    print(crm)
    best_cr["accuracy"]=crm["accuracy"]
    best_cr["macro avg"]=crm["macro_avg"]
    best_cr["macro avg"]["support"]=crm["support"]
    best_cr["weighted avg"]=crm["weighted_avg"]
    best_cr["weighted avg"]["support"]=crm["support"]
    global cm
    global cr
    print(best_cr)
    print(best_cm)
    cm={}
    cr=[]

@app.route("/analytics")
def analytics():
    table_head=["version_name","notes","trained_on_samples","creates_at"]
    table_data=[]
    version_names=[]
    version_data = requests.get(
        "http://127.0.0.1:5001/mlmodel/getModelVersion"
    ).json()
    for data in version_data["model_versions"]:
        table_data.append([
            data["version_name"],
            data["notes"],
            data["trained_on_samples"],
            data["created_at"]
            ])
        version_names.append(data["version_name"])

    y_id=[]
    y_accuracy=[]
    y_worst_f1=[]
    y_worst_cls=[]
    y_misclassification=[]
    for version_id in version_names:
        response = requests.post(
            "http://127.0.0.1:5001/mlmodel/getVersionMetrics",
            json={"model_version_id": version_id}
        ).json()
        data=response["version_summary"]
        if data != None:
            y_id.append(data["model_version_id"])
            y_accuracy.append(data["accuracy"])
            y_worst_f1.append(data["worst_class_f1"])
            y_worst_cls.append(data["worst_class_name"])
            y_misclassification.append(data["misclassification_rate"])

    fig=go.Figure()

    #accuracy_line
    fig.add_trace(go.Scatter(
        x=version_names,
        y=y_accuracy,
        mode="lines+markers",
        name="Accuracy",
        line=dict(color="green"),
        hovertemplate="Version: %{x}<br>Accuracy: %{y:.4f}<extra></extra>"
    ))
    
    #worst f1
    fig.add_trace(go.Scatter(
        x=version_names,
        y=y_worst_f1,
        mode="lines+markers",
        name="worst class f1",
        customdata=y_worst_cls,
        line=dict(color="yellow"),
        hovertemplate=(
            "Version: %{x}<br>"
            "Worst Class: %{customdata}<br>"
            "F1 Score: %{y:.4f}"
            "<extra></extra>"
        )        
    ))

    print(y_misclassification)
    # Misclassification line
    fig.add_trace(go.Scatter(
        x=version_names,
        y=y_misclassification,
        mode="lines+markers",
        name="No of correct classifications",
        line=dict(color="blue"),
        hovertemplate="Version: %{x}<br>Misclassification: %{y:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title="Model Version Performance",
        xaxis_title="Model Version",
        yaxis_title="Metric Value",
        dragmode="pan",
        hovermode="closest"
    )

    graph_html = fig.to_html(full_html=False)

    return render_template("dashboard.html", table_head=table_head, table_data=table_data, graph_html=graph_html, version_names=version_names)

def wipeAllMetrics():
    response=requests.post("http://127.0.0.1:5001/mlmodel/wipeAllMetrics")
    print(response.json())

@app.route("/changedCls",methods=["POST"])
def changedCls():
    req_data=request.get_json()
    new_cls=req_data["changed_cls"]
    global class_names
    class_names=new_cls
    global class_count
    class_count=len(class_names)
    wipeAllMetrics()
    global best_cm;global best_cr
    global cm;global cr
    print({"New class set":new_cls})
    best_cm=[]
    best_cr={}
    cm=best_cm
    cr=best_cr
    return redirect(url_for("threshold"))

@app.route("/getModelData",methods=["POST"])
def getModelData():
    model_version=request.form.get("model_version")
    # model level trace
    response= requests.post(
        "http://127.0.0.1:5001/mlmodel/getModelLevelTrace",
        json={"model_version_id": model_version}
    ).json()
    data=response["model_level_trace"]
    model_level_metrics=[
        [data["accuracy"]],
        ["f1-score","recall","support","precision"],
        [data["macro_f1"],data["macro_recall"],data["macro_support"],data["macro_precision"]],
        [data["weighted_f1"],data["weighted_recall"],data["weighted_support"],data["weighted_precision"]]
    ]
    
    # class level trace
    response = requests.post(
        "http://127.0.0.1:5001/mlmodel/getClassLevelTrace",
        json={"model_version_id": model_version}
    ).json()
    data=response["class_level_trace"]
    class_level_metrics_head=["class name","recall","f1 score","support"]
    class_level_metrics=[[row["class_name"],row["recall"],row["f1_score"],row["support"]] for row in data]

    response = requests.post(
        "http://127.0.0.1:5001/mlmodel/getErrorLevelTrace",
        json={"model_version_id": model_version}
    ).json()
    data=response["error_level_trace"]
    error_level_metrics=[row["count"] for row in data]
    pointer=0
    global class_count
    confusion_matrix=[[0 for _ in range(class_count)]for _ in range(class_count)]
    for i in range(class_count):
        for j in range(class_count):
            confusion_matrix[i][j]=error_level_metrics[pointer]
            pointer+=1
    global class_names
    return render_template("analytics.html",model_version=model_version,model_level_metrics=model_level_metrics,class_level_metrics_head=class_level_metrics_head,class_level_metrics=class_level_metrics,confusion_matrix=confusion_matrix,class_names=class_names)

init_metrics()
if __name__=="__main__":
    app.run(debug=True,port=5002)

#things to do later 
# - adding required to html
# - post() generalisation not necessary 
# - any type of file upload for ftp + make it work for any column name by getting input