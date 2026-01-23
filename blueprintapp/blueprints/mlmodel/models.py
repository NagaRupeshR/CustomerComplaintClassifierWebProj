from blueprintapp.app import db

class Class_name(db.Model):
    __tablename__="class_names"
    
    id=db.Column(db.Integer,primary_key=True)
    cname=db.Column(db.Text,nullable=False)

    def __repr__(self):
        return f"class name : {self.cname}"

class ClassMetric(db.Model):
    __tablename__ = "class_metrics"

    id = db.Column(db.Integer, primary_key=True)
    class_name = db.Column(db.Text, nullable=False, unique=True)

    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    support = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        return {
            "class_name": self.class_name,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "support": self.support
        }


class ModelMetric(db.Model):
    __tablename__ = "model_metrics"

    id = db.Column(db.Integer, primary_key=True)

    accuracy = db.Column(db.Float, nullable=False)

    macro_precision = db.Column(db.Float, nullable=False)
    macro_recall = db.Column(db.Float, nullable=False)
    macro_f1 = db.Column(db.Float, nullable=False)

    weighted_precision = db.Column(db.Float, nullable=False)
    weighted_recall = db.Column(db.Float, nullable=False)
    weighted_f1 = db.Column(db.Float, nullable=False)

    total_support = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        return {
            "accuracy": self.accuracy,
            "macro_avg": {
                "precision": self.macro_precision,
                "recall": self.macro_recall,
                "f1_score": self.macro_f1
            },
            "weighted_avg": {
                "precision": self.weighted_precision,
                "recall": self.weighted_recall,
                "f1_score": self.weighted_f1
            },
            "support": self.total_support
        }
    
class ConfusionMatrix(db.Model):
    __tablename__="confusion_matrix"
    
    id=db.Column(db.Integer,primary_key=True)
    pred_cls=db.Column(db.Text,nullable=False)
    actual_cls=db.Column(db.Text,nullable=False)
    classification_count=db.Column(db.Integer,nullable=False)

    def to_dict(self):
        return {
            "id":self.id,
            "predicted_class":self.pred_cls,
            "actual_class":self.actual_cls,
            "ClassificationCount":self.classification_count
        }
    
class Misclassified(db.Model):
    __tablename__="misclassified_data"

    id=db.Column(db.Integer,primary_key=True)
    complaint=db.Column(db.Text,nullable=False)
    actual_cls=db.Column(db.Text,nullable=False)
    pred_cls=db.Column(db.Text,nullable=False)

    def to_dict(self):
        return {
            "id":self.id,
            "complaint":self.complaint,
            "actual_cls":self.actual_cls,
            "pred_cls":self.pred_cls
        }

class SettingVars(db.Model):
    __tablename__="setting_variables"

    id=db.Column(db.Integer,primary_key=True)
    threshold=db.Column(db.Integer,nullable=False)
    ignore_misclassification=db.Column(db.Boolean,nullable=False,default=False)

    def to_dict(self):
        return {
            "id":self.id,
            "threshold":self.threshold,
            "ignore_misclassification":self.ignore_misclassification
        }
    
class ModelVersion(db.Model):
    __tablename__="model_versions"

    version_name=db.Column(db.Text,primary_key=True)
    notes=db.Column(db.Text)
    trained_on_samples=db.Column(db.Integer,nullable=False)
    created_at=db.Column(db.DateTime,server_default=db.func.now(),nullable=False)

    model_metrics=db.relationship(
        "ModelLevelTracker",backref="model_version",uselist=False,cascade="all, delete-orphan"
    )
    class_metrics=db.relationship(
        "ClassLevelTracker",backref="model_version",cascade="all, delete-orphan"
    )
    error_metrics=db.relationship(
        "ErrorLevelTracker",backref="model_version",cascade="all, delete-orphan"
    )
    version_summary_metrics=db.relationship(
        "VersionSummaryMetrics",backref="model_version",uselist=False,cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "version_name": self.version_name,
            "notes": self.notes,
            "trained_on_samples": self.trained_on_samples,
            "created_at": self.created_at.isoformat()        
        }

class ModelLevelTracker(db.Model):
    __tablename__ = "model_level_trackers"

    id = db.Column(db.Integer, primary_key=True)
    model_version_id = db.Column(
        db.Text,
        db.ForeignKey("model_versions.version_name", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )

    accuracy = db.Column(db.Float, nullable=False)

    macro_precision = db.Column(db.Float, nullable=False)
    macro_recall = db.Column(db.Float, nullable=False)
    macro_f1 = db.Column(db.Float, nullable=False)
    macro_support = db.Column(db.Float, nullable=False)

    weighted_precision = db.Column(db.Float, nullable=False)
    weighted_recall = db.Column(db.Float, nullable=False)
    weighted_f1 = db.Column(db.Float, nullable=False)
    weighted_support = db.Column(db.Float, nullable=False)

    def to_dict(self):
        return {
            "model_version_id": self.model_version_id,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "macro_recall":self.macro_recall,
            "macro_support":self.macro_support,
            "macro_precision":self.macro_precision,
            "weighted_f1": self.weighted_f1,
            "weighted_recall":self.weighted_recall,
            "weighted_support":self.weighted_support,
            "weighted_precision":self.weighted_precision
        }

class ClassLevelTracker(db.Model):
    __tablename__ = "class_level_trackers"

    id = db.Column(db.Integer, primary_key=True)
    model_version_id = db.Column(
        db.Text,
        db.ForeignKey("model_versions.version_name", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    class_name = db.Column(db.Text, nullable=False)

    recall = db.Column(db.Float, nullable=False)
    f1 = db.Column(db.Float, nullable=False)
    support = db.Column(db.Integer, nullable=False)

    __table_args__ = (
        db.UniqueConstraint(
            "model_version_id", "class_name",
            name="uq_class_metric_per_version"
        ),
    )

    def to_dict(self):
        return {
            "model_version_id": self.model_version_id,
            "class_name": self.class_name,
            "recall": self.recall,
            "f1_score": self.f1,
            "support": self.support
        }

class ErrorLevelTracker(db.Model):
    __tablename__ = "error_level_trackers"

    id = db.Column(db.Integer, primary_key=True)
    model_version_id = db.Column(
        db.Text,
        db.ForeignKey("model_versions.version_name", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    predicted_class = db.Column(db.Text, nullable=False)
    actual_class = db.Column(db.Text, nullable=False)
    classification_count = db.Column(db.Integer, nullable=False)

    __table_args__ = (
        db.UniqueConstraint(
            "model_version_id", "predicted_class", "actual_class",
            name="uq_confusion_matrix_cell"
        ),
    )

    def to_dict(self):
        return {
            "model_version_id": self.model_version_id,
            "predicted_class": self.predicted_class,
            "actual_class": self.actual_class,
            "count": self.classification_count
        }

class VersionSummaryMetrics(db.Model):
    __tablename__ = "version_summary_metrics"

    id = db.Column(db.Integer, primary_key=True)
    model_version_id = db.Column(
        db.Text,
        db.ForeignKey("model_versions.version_name", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )

    accuracy = db.Column(db.Float, nullable=False)
    worst_class_f1 = db.Column(db.Float, nullable=False)
    worst_class_name = db.Column(db.Text, nullable=False)
    misclassification_rate = db.Column(db.Float, nullable=False)

    def to_dict(self):
        return {
            "model_version_id": self.model_version_id,
            "accuracy": self.accuracy,
            "worst_class_f1": self.worst_class_f1,
            "worst_class_name": self.worst_class_name,
            "misclassification_rate": self.misclassification_rate
        }
