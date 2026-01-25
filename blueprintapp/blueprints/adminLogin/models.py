from flask_login import UserMixin
from blueprintapp.app import db

class Admin(db.Model,UserMixin):
    __tablename__="admin"

    uid=db.Column(db.Integer,primary_key=True)
    username=db.Column(db.Text,nullable=False,unique=True)
    password=db.Column(db.Text,nullable=False,unique=True)

    def to_dict(self):
        return {"username":self.username,"password":self.password}