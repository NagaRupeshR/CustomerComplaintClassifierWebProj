from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db=SQLAlchemy()

def create_app():
    app=Flask(__name__,template_folder="templates")
    app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///./testdb.db'
    db.init_app(app)

    #import and register all blueprints
    from blueprintapp.blueprints.mlmodel.routes import mlmodel
    app.register_blueprint(mlmodel,url_prefix='/mlmodel')

    migrate=Migrate(app,db)

    return app