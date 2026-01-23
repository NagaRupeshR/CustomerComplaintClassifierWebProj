import joblib
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix
import numpy as np
from flask import Flask,request, jsonify

app=Flask(__name__,template_folder="templates",static_folder="static",static_url_path="/")


# Load Saved Model, TF-IDF, LabelEncoder
model = joblib.load("./models/bestModels/model.pkl")
tfidf = joblib.load("./models/bestModels/tfidf.pkl")
label_encoder = joblib.load("./models/bestModels/label_encoder.pkl")

print("Loading SBERT model...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Rule-based Adjustment Layer (same function as before)
def rule_based_adjustment(text, predicted_label):
    t = text.lower()
    upi_keywords = ["upi", "vpa", "virtual payment address","qr", "gpay", "phonepe", "paytm"]
    retail_keywords = ["savings account", "passbook", "atm", "debit card", "kyc", "ifsc"]
    has_upi = any(k in t for k in upi_keywords)
    has_retail = any(k in t for k in retail_keywords)
    if has_upi:
        return "upi_transaction_failures"
    if (not has_upi) and has_retail and predicted_label == "upi_transaction_failures":
        return "retail_banking"
    return predicted_label

@app.route("/predict_complaint",methods=["POST"])
def predict_complaint():
    data=request.get_json()
    text=data.get('complaint')
    # Encode & combine features
    sbert_emb = csr_matrix(sbert_model.encode([text]))
    tfidf_vec = tfidf.transform([text])
    combined = hstack([tfidf_vec, sbert_emb])

    # Predict class index
    pred_prob = model.predict_proba(combined)[0]  # probability for each class
    pred_indices = np.argsort(pred_prob)[::-1]    # sort high → low

    # Get class names and probabilities
    class_names = label_encoder.inverse_transform(pred_indices)
    class_probs = [float(pred_prob[i]) for i in pred_indices]

    return jsonify({
        "predicted":[[class_names[i],class_probs[i]] for i in range(6)]
        })

if __name__=="__main__":
    app.run(port=5000)