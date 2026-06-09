
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import pickle, json, os, numpy as np

# ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# # Fetch latest version of your registered model
# registered_model = ml_client.models.get(
#     name="knn_diabetes_model",   # the name you used when registering
#     version="6"                  # or omit version to get latest
# )

print(f"Using model: {registered_model.name}, version {registered_model.version}")
def init():
    global model
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model.pkl")
    model = pickle.load(open(model_path, "rb"))

def run(raw_data):
    data = json.loads(raw_data)
    features = [float(data["Glucose"]), float(data["BMI"]), float(data["Age"])]
    prediction = model.predict([features])
    confidence = model.predict_proba([features])
    return {
        "prediction": int(prediction[0]),
        "confidence": round(float(np.max(confidence)) * 100, 2)
    }