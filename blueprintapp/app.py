from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_bcrypt import Bcrypt

db=SQLAlchemy()

def create_app():
    app=Flask(__name__,template_folder="templates")
    app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///./testdb.db'
    app.secret_key="Mankatha alavu illa brooo"

    db.init_app(app)

    #import and register all blueprints
    from blueprintapp.blueprints.mlmodel.routes import mlmodel
    app.register_blueprint(mlmodel,url_prefix='/mlmodel')

    from blueprintapp.blueprints.customerLogin.routes import customerLogin
    app.register_blueprint(customerLogin,url_prefix='/customerLogin')

    from blueprintapp.blueprints.adminLogin.routes import adminLogin
    app.register_blueprint(adminLogin,url_prefix='/adminLogin')

    migrate=Migrate(app,db)

    return app