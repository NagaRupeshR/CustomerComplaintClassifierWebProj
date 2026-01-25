from flask import jsonify, Blueprint, request
from blueprintapp.app import db
from blueprintapp.blueprints.adminLogin.models import Admin

adminLogin=Blueprint('adminLogin',__name__,template_folder="templates")

@adminLogin.route("/setAuth",methods=["POST"])
def setAuth():
    return ""
@adminLogin.route("/getAuth",methods=["POST"])
def getAuth():
    return ""