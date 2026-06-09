import requests, json

#endpoint_info = ml_client.online_endpoints.get("diabetes-endpoint")
#key = ml_client.online_endpoints.get_keys("diabetes-endpoint").primary_key
url=""
api_key=""
headers={
    "content-Type":"application/json",
    "Authorization":f"Bearer {api_key}"
}

response = requests.post(
    endpoint_info.scoring_uri,
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    data=json.dumps({"Glucose": 148, "BMI": 33.6, "Age": 50})
)
print(response.json())
# → {"prediction": 1, "confidence": 74.5}