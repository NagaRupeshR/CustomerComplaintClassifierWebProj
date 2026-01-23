from flask import Flask,render_template,request,redirect,url_for,jsonify
from flask import session
import requests

app=Flask(__name__,template_folder="templates",static_folder="static",static_url_path="/")

app.secret_key="popK3Qrtv47763rtezf49oppo"

threshold_misclassification=10

#session code ---------------------------------------------

# @app.route("/getSessionCookie",methods=["POST"])
def setSessionData(complaint,pred_cls):
    session['current_complaint']=complaint
    session['pred_cls']=pred_cls

def getSessionData():
    return {"current_complaint":session.get("current_complaint"),"pred_cls":session.get("pred_cls")}

def clearSession():
    session.clear()

#customer side --------------------------------------------

@app.route("/customer")
def index():
    return render_template("index.html",complaint="",predicted=[])

@app.route("/Customerclassify", methods=["POST"])
def customerclassify():
    complaint = request.form.get("complaint")
    url = "http://127.0.0.1:5000/predict_complaint"
    req_data={"complaint":complaint}
    response=requests.post(url,json=req_data)
    predicted=response.json()["predicted"]
    setSessionData(complaint,predicted[0][0])
    return render_template("index.html",complaint=complaint,predicted=predicted)

@app.route("/misclassification",methods=["POST"])
def misclassification():
    expectedCls=request.form.get('expectedCls')
    sessionData=getSessionData()
    req_data={"complaint":sessionData['current_complaint'],"actual_cls":expectedCls,"pred_cls":sessionData['pred_cls']}
    clearSession()
    url = "http://127.0.0.1:5001/mlmodel/setMisclassification"
    response=requests.post(url,json=req_data)
    global threshold_misclassification
    if response.json()["count"]>threshold_misclassification:#--------------
        return redirect("http://127.0.0.1:5002/threshold")
    return redirect(url_for("index"))

# threshold management with minimal overhead concern
# it is necessary to get frequenty changing data like count and complaint
# it would not be effficient if i did make a post request to the database for threshold that changes rarely
# so every time this .py running machine starts it fetched the threshold
# the very next chance it changes is when the admin.py changes it
# admin informs about that just by a single request to changedThrehold 
# so knocking out all the request per classification misclassification

@app.route("/changedThreshold",methods=["POST"])
def changedThreshold():
    th=request.get_json()['threshold']
    global threshold_misclassification
    threshold_misclassification=int(th)
    print("New threshold Set",threshold_misclassification)
    return jsonify({"response":"heyy the threshold got updated @ customer side"})

def getThreshold():
    url="http://127.0.0.1:5001/mlmodel/getThreshold"
    response=requests.post(url)
    global threshold_misclassification
    threshold_misclassification=int(response.json()['threshold'])
    print("Initial threshold Set",threshold_misclassification)
getThreshold()
if __name__=="__main__":
    app.run(debug=True,port=8000)