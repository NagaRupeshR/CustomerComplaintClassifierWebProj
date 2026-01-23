from flask import Flask,render_template,request,send_file,redirect,url_for, Response
import pandas as pd
import io
import requests

app=Flask(__name__,template_folder="templates",static_folder="static",static_url_path="/")

threshold_misclassification=10
ignoreMisclassification=False

@app.route("/")
def adminIndex():
    return render_template("admin.html",complaint="",threshold=threshold_misclassification)

@app.route("/ignoreMisCls",methods=["POST"])
def ignoreMisCls():
    ignore=request.form.get("ignore")
    global ignoreMisclassification
    ignoreMisclassification=(ignore=="1")
    url="http://127.0.0.1:5001/mlmodel/setIgnore"
    response=requests.post(url,json={"ignore":ignoreMisclassification})
    print(response)
    return redirect(url_for('adminIndex'))

@app.route("/fetchThreshold",methods=["POST"])
def fetch_threshold():
    threshold=request.form.get("threshold")
    global threshold_misclassification
    threshold_misclassification=int(threshold)
    url="http://127.0.0.1:5001/mlmodel/setThreshold"
    response=requests.post(url,json={"threshold":threshold_misclassification})
    url="http://127.0.0.1:8000/changedThreshold"
    response=requests.post(url,json={"threshold":threshold_misclassification})
    print(response.json())
    return redirect(url_for("adminIndex"))

@app.route("/classify",methods=["POST"])
def classify():
    complaint=request.form.get("complaint")
    url = "http://127.0.0.1:5000/predict_complaint"
    req_data={"complaint":complaint}
    response=requests.post(url,json=req_data)
    predicted=response.json()["predicted"]
    return render_template("admin.html",complaint=complaint,predicted=predicted,threshold=threshold_misclassification)

@app.route("/setClass",methods=["POST"])
def setClass():
    cls=request.form.get("cls")
    cls=cls.split()
    url = "http://127.0.0.1:5001/mlmodel/setClasses"
    req_data={"classes":cls}
    response=requests.post(url,json=req_data)
    print(response.json()["message"])
    
    url="http://127.0.0.1:5002/changedCls"
    response=requests.post(url,json={"changed_cls":cls})
    print("informed cls change to train_model.py")

    return redirect(url_for('adminIndex'))

@app.route("/bulk",methods=["GET","POST"]) # this is only for analysis 
def bulk():
    if request.method=="GET":
        return render_template("bulk.html",html_fragment="<p></p>")
    if request.method=="POST":
        download_mode=request.form.get("download_mode")
        file=request.files['file']
        df=pd.read_csv(file)
        bulkPredictionClass=[]
        bulkPredictionProb=[]
        bulkPredictionClass2=[]
        bulkPredictionProb2=[]
        for index,row in df.iterrows():
            complaint=row['complaint']
            url = "http://127.0.0.1:5000/predict_complaint"
            req_data={"complaint":complaint}
            response=requests.post(url,json=req_data)
            pred=response.json()["predicted"]
            bulkPredictionClass.append(pred[0][0])
            bulkPredictionProb.append(pred[0][1])
            bulkPredictionClass2.append(pred[1][0])
            bulkPredictionProb2.append(pred[1][1])
            global ignoreMisclassification
            print(ignoreMisclassification,row['actualcls'],pred[0][0])
            if not ignoreMisclassification and row['actualcls']!=pred[0][0]:
                url = "http://127.0.0.1:5001/mlmodel/setMisclassification"
                req_data={"complaint":complaint,"actual_cls":row['actualcls'],"pred_cls":pred[0][0]}
                response=requests.post(url,json=req_data)
        df['1stcls']=bulkPredictionClass
        df['p(1stcls)']=bulkPredictionProb
        df['2ndcls']=bulkPredictionClass2
        df['p(2ndcls)']=bulkPredictionProb2
        if download_mode=="without_download":
            isgtThres=True #some value
            url="http://127.0.0.1:5001/mlmodel/getMisclsCount"
            response=requests.post(url)
            count=response.json()['count']
            global threshold_misclassification
            if count>threshold_misclassification:
                isgtThres=True
            #global ignoreMisclassification
            isgtThres=ignoreMisclassification==False
            return render_template("bulk.html",html_fragment=df.to_html(),isgtThres=isgtThres)
        if download_mode=="with_download":
            buffer=io.StringIO()
            df.to_csv(buffer,index=False)
            buffer.seek(0)
            return send_file(
                io.BytesIO(buffer.getvalue().encode("utf-8")),
                mimetype="text/csv",
                as_attachment=True,
                download_name="bulk_processed.csv"
            )
        
def getSettingsFromDB():
    url="http://127.0.0.1:5001/mlmodel/getSettings"
    response=requests.post(url)
    global threshold_misclassification
    global ignoreMisclassification
    threshold_misclassification=response.json()['settings']['threshold']
    ignoreMisclassification=response.json()['settings']['ignore_misclassification']

getSettingsFromDB()

@app.route("/action",methods=["POST"])#takes to the threshold.html for ACTION
def action():
    return redirect("http://127.0.0.1:5002/threshold")

@app.route("/exportMisCls",methods=["POST"])
def exportMisCls():
    url="http://127.0.0.1:5001/mlmodel/getMisclassification"
    response=requests.post(url)
    data=response.json()["misclassified"]
    df=pd.DataFrame(data)
    buffer=io.StringIO()
    df.to_csv(buffer,index=True)
    buffer.seek(0)
    return send_file(
        io.BytesIO(buffer.getvalue().encode("utf-8")),
        mimetype='text/csv',
        as_attachment=True,
        download_name="misclassified.csv"
    )

if __name__=="__main__":
    app.run(debug=True,port=8080)
    