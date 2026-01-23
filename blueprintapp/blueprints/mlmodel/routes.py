from flask import jsonify, Blueprint, request
from blueprintapp.blueprints.mlmodel.models import Class_name,ClassMetric, ModelMetric, ConfusionMatrix, Misclassified, SettingVars, ModelVersion, ModelLevelTracker, ClassLevelTracker, ErrorLevelTracker, VersionSummaryMetrics
from blueprintapp.app import db

#this was done for a blueprint based model in general flask wraps all routes into a function to avoid circular imports

mlmodel=Blueprint('mlmodel',__name__,template_folder="templates")

@mlmodel.route("/getClasses",methods=["POST","GET"])
def getClasses():
    classes = Class_name.query.all()
    result = [c.cname for c in classes]
    return jsonify({"classes": result})

@mlmodel.route("/setClasses", methods=["POST","GET"]) # remove get later for all sets
def setClasses(): #make dynamic class get using admin side class changes
    Class_name.query.delete()
    db.session.commit()
    classes = request.get_json()['classes']

    for cls in classes:
        exists = Class_name.query.filter_by(cname=cls).first()
        if not exists:
            db.session.add(Class_name(cname=cls))

    db.session.commit()
    return jsonify({"message":"Classes set"})

@mlmodel.route('/getClsCount',methods=["POST"])
def getClsCount():
    return jsonify({"ClassCount":Class_name.query.count()})

@mlmodel.route("/setMetrics",methods=["POST","GET"])
def setMetrics():
    data=request.get_json()
    ClassMetric.query.delete()
    db.session.commit()
    ModelMetric.query.delete()
    db.session.commit()
    for key,value in data.items():
        if key in ("accuracy","macro avg","weighted avg"):
            continue
        metric=ClassMetric.query.filter_by(class_name=key).first()
        
        if not metric:
            metric=ClassMetric(class_name=key)
        metric.precision = float(value["precision"])
        metric.recall = float(value["recall"])
        metric.f1_score = float(value["f1-score"])
        metric.support = int(value["support"])

        db.session.add(metric)
    overall = ModelMetric.query.first()
    if not overall:
        overall = ModelMetric(
            accuracy=float(data["accuracy"]),
            macro_precision=float(data["macro avg"]["precision"]),
            macro_recall=float(data["macro avg"]["recall"]),
            macro_f1=float(data["macro avg"]["f1-score"]),
            weighted_precision=float(data["weighted avg"]["precision"]),
            weighted_recall=float(data["weighted avg"]["recall"]),
            weighted_f1=float(data["weighted avg"]["f1-score"]),
            total_support=int(data["macro avg"]["support"])
        )
        db.session.add(overall)
    else:
        overall.accuracy = float(data["accuracy"])
        overall.macro_precision = float(data["macro avg"]["precision"])
        overall.macro_recall = float(data["macro avg"]["recall"])
        overall.macro_f1 = float(data["macro avg"]["f1-score"])
        overall.weighted_precision = float(data["weighted avg"]["precision"])
        overall.weighted_recall = float(data["weighted avg"]["recall"])
        overall.weighted_f1 = float(data["weighted avg"]["f1-score"])
        overall.total_support = int(data["macro avg"]["support"])
    db.session.commit()
    return jsonify({"message": "Metrics saved successfully"})

@mlmodel.route("/metrics/classes",methods=["POST","GET"])
def get_class_metrics():
    metrics = ClassMetric.query.all()
    return jsonify({"class_metrics":[m.to_dict() for m in metrics]})


@mlmodel.route("/metrics/model",methods=["POST","GET"])
def get_model_metrics():
    metric = ModelMetric.query.first()
    return jsonify({"model_metrics":metric.to_dict() if metric else {}})

@mlmodel.route("/setConfusionMatrix",methods=["POST","GET"])
def setConfusionMatrix():
    req_data=request.get_json()['confusion_matrix']
    ConfusionMatrix.query.delete()
    db.session.commit()
    classes = Class_name.query.all()
    cm=req_data
    n=len(classes)
    for i in range(n):
        for j in range(n):
            row=ConfusionMatrix(pred_cls=classes[i].cname,actual_cls=classes[j].cname,classification_count=cm[i][j])
            db.session.add(row)
    db.session.commit()
    return jsonify({"message":"Confusion Matrix SET"})

@mlmodel.route("/getConfusionMatrix",methods=["POST","GET"])
def getConfusionMatrix():
    cm=ConfusionMatrix.query.all()
    result = [row.to_dict() for row in cm]
    return jsonify({"confusion_matrix":result})

@mlmodel.route("/setMisclassification",methods=["POST","GET"])
def setMisclassification():
    req=request.get_json()
    row=Misclassified(complaint=req["complaint"],actual_cls=req["actual_cls"],pred_cls=req["pred_cls"])
    db.session.add(row)
    db.session.commit()
    return jsonify({"count":Misclassified.query.count()})

@mlmodel.route("/getMisclassification",methods=["POST","GET"])
def getMisclassification():
    misclasses=Misclassified.query.all()
    result=[row.to_dict() for row in misclasses]
    return jsonify({"misclassified":result})

@mlmodel.route("/delMisclses",methods=["POST","GET"])
def delMisclses():
    Misclassified.query.delete()
    db.session.commit()
    return jsonify({"response":"table truncated"})

@mlmodel.route("/getMisclsCount",methods=["GET","POST"])
def getMisclsCount():
    return jsonify({"count":Misclassified.query.count()})

@mlmodel.route("/setThreshold",methods=["POST","GET"])
def setThreshold():
    th=request.get_json()['threshold']
    settings = SettingVars.query.first()
    if not settings:
        row=SettingVars(threshold=th)
        db.session.add(row)
        settings = SettingVars.query.first()
    settings.threshold = th
    db.session.commit()
    return "Threshold SET"

@mlmodel.route("/setIgnore",methods=["POST","GET"])
def setIgnore():
    ignore=request.get_json()['ignore']
    settings = SettingVars.query.first()
    if not settings:
        row=SettingVars(threshold=10)
        db.session.add(row)
        settings = SettingVars.query.first()
    settings.ignore_misclassification=ignore
    db.session.commit()
    return "Ignore misclassification is ASSIGNED"

@mlmodel.route("/getSettings",methods=["POST","GET"])
def getSettings():
    settings = SettingVars.query.first()
    if not settings:
        return "No settings found", 404
    return jsonify({"settings":settings.to_dict()})

@mlmodel.route("/getThreshold",methods=["POST","GET"])
def getThreshold():
    settings=SettingVars.query.first()
    if not settings:
        return "No settings found", 404
    return jsonify({"threshold":settings.to_dict()['threshold']})

@mlmodel.route("/setModelVersion", methods=["POST"])
def setModelVersion():
    data = request.get_json()["model_version"]

    row = ModelVersion(
        version_name=data["version_name"],
        notes=data.get("notes", ""),
        trained_on_samples=data["trained_on_samples"]
    )

    db.session.add(row)
    db.session.commit()

    return jsonify({"new_model_version": row.to_dict()})


@mlmodel.route("/setModelLevelTrace", methods=["POST"])
def setModelLevelTrace():
    data = request.get_json()["mlt"]

    # enforce uniqueness
    ModelLevelTracker.query.filter_by(
        model_version_id=data["model_version_id"]
    ).delete()

    row = ModelLevelTracker(
        model_version_id=data["model_version_id"],
        accuracy=data["accuracy"],
        macro_f1=data["macro_f1"],
        macro_recall=data["macro_recall"],
        macro_support=data["macro_support"],
        macro_precision=data["macro_precision"],
        weighted_f1=data["weighted_f1"],
        weighted_recall=data["weighted_recall"],
        weighted_support=data["weighted_support"],
        weighted_precision=data["weighted_precision"]
    )

    db.session.add(row)
    db.session.commit()

    return jsonify({"model_level_trace_set": row.to_dict()})

@mlmodel.route("/setClassLevelTrace", methods=["POST"])
def setClassLevelTrace():
    rows = request.get_json()["clt"]
    version_id = rows[0]["model_version_id"]

    # enforce uniqueness per version
    ClassLevelTracker.query.filter_by(
        model_version_id=version_id
    ).delete()

    for row_data in rows:
        row = ClassLevelTracker(
            model_version_id=row_data["model_version_id"],
            class_name=row_data["class_name"],
            recall=row_data["recall"],
            f1=row_data["f1"],
            support=row_data["support"]
        )
        db.session.add(row)

    db.session.commit()
    return jsonify({"message": "class level trace set"})

@mlmodel.route("/setErrorLevelTrace", methods=["POST"])
def setErrorLevelTrace():
    data = request.get_json()["elt"]

    ErrorLevelTracker.query.filter_by(
        model_version_id=data["model_version_id"]
    ).delete()

    for cell in data["matrix"]:
        row = ErrorLevelTracker(
            model_version_id=data["model_version_id"],
            predicted_class=cell["predicted_class"],
            actual_class=cell["actual_class"],
            classification_count=cell["count"]
        )
        db.session.add(row)

    db.session.commit()
    return jsonify({"message": "error level trace set"})


@mlmodel.route("/setVersionMetrics", methods=["POST"])
def setVersionMetrics():
    data = request.get_json()["version_summary"]

    VersionSummaryMetrics.query.filter_by(
        model_version_id=data["model_version_id"]
    ).delete()

    row = VersionSummaryMetrics(
        model_version_id=data["model_version_id"],
        accuracy=data["accuracy"],
        worst_class_f1=data["worst_class_f1"],
        worst_class_name=data["worst_class_name"],
        misclassification_rate=data["misclassification_rate"]
    )

    db.session.add(row)
    db.session.commit()

    return jsonify({"message": "version summary set"})

@mlmodel.route("/getModelVersion", methods=["GET"])
def getModelVersion():
    data = ModelVersion.query.all()
    return jsonify({"model_versions": [row.to_dict() for row in data]})

@mlmodel.route("/getModelLevelTrace", methods=["POST"])
def getModelLevelTrace():
    version_id = request.get_json()["model_version_id"]

    row = ModelLevelTracker.query.filter_by(
        model_version_id=version_id
    ).first()

    return jsonify({"model_level_trace": row.to_dict() if row else None})

@mlmodel.route("/getClassLevelTrace", methods=["POST"])
def getClassLevelTrace():
    version_id = request.get_json()["model_version_id"]

    data = ClassLevelTracker.query.filter_by(
        model_version_id=version_id
    ).all()

    return jsonify({"class_level_trace": [row.to_dict() for row in data]})

@mlmodel.route("/getErrorLevelTrace", methods=["POST"])
def getErrorLevelTrace():
    version_id = request.get_json()["model_version_id"]

    data = ErrorLevelTracker.query.filter_by(
        model_version_id=version_id
    ).all()

    return jsonify({"error_level_trace": [row.to_dict() for row in data]})

@mlmodel.route("/getVersionMetrics", methods=["POST"])
def getVersionMetrics():
    version_id = request.get_json()["model_version_id"]

    row = VersionSummaryMetrics.query.filter_by(
        model_version_id=version_id
    ).first()

    return jsonify({"version_summary": row.to_dict() if row else None})

@mlmodel.route("/wipeAllMetrics", methods=["GET","POST"])
def wipeAllMetrics():
    # child tables first
    ErrorLevelTracker.query.delete()
    ClassLevelTracker.query.delete()
    ModelLevelTracker.query.delete()
    VersionSummaryMetrics.query.delete()

    # parent table last
    ModelVersion.query.delete()

    db.session.commit()

    return jsonify({
        "message": "ALL TABLES CLEARED 🔥",
        "status": "success"
    })

