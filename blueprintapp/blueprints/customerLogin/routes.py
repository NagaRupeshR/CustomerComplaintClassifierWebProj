from flask import jsonify, Blueprint, request
from blueprintapp.app import db
from blueprintapp.blueprints.customerLogin.models import Customer

customerLogin=Blueprint('customerLogin',__name__,template_folder="templates")

@customerLogin.route("/setAuth",methods=["POST"])
def setAuth():
    req_data=request.get_json()
    username=req_data["username"]
    password=req_data["password"]
    
    row=Customer(username=username,password=password)
    db.session.add(row)
    db.session.commit()
    return jsonify(row.to_dict())

@customerLogin.route("/getAuth",methods=["POST"])
def getAuth():
    req_data=request.get_json()
    username=req_data["username"]
    password=req_data["password"]

    row = Customer.query.filter_by(
        username=username
    ).first()
    if row=={}:
        return jsonify({"message":"not yet authenticated"})
    if row.to_dict()["password"]==password:
        return jsonify({"message":"Authenticated"})
    else:
        return jsonify({"message":"incorrect password"})