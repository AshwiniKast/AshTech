from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    Environment,
    CodeConfiguration
)
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import pickle, json, os, numpy as np
import uuid
from azure.identity import DeviceCodeCredential
credential = DeviceCodeCredential(tenant_id="" )

ml_client = MLClient.from_config(credential=DefaultAzureCredential(), path="/config.json")

print(ml_client.subscription_id)   # should now show real values
print(ml_client.workspace_name)
ws = ml_client.workspaces.get("MyWorkspace1")
print("Workspace location:", ws.location)
print("✅ Connected successfully!")

# Fetch latest version of your registered model
registered_model = ml_client.models.get(
    name="knn_diabetes_model",   # the name you used when registering
    version="6"                  # or omit version to get latest
)

print(f"Using model: {registered_model.name}, version {registered_model.version}")
endpoint_name = f"diabetes-ep-{str(uuid.uuid4())[:8]}"

# 1. Create endpoint
endpoint = ManagedOnlineEndpoint(
    name= endpoint_name, #"diabetes-endpoint",
    auth_mode="key"
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Endpoint created!")

# 2. Define environment
# env = Environment(
#     image="mcr.microsoft.com/azureml/openmini4.1.0-ubuntu20.04",
#     conda_file="conda.yaml",
#     name="diabetes-env"
#)
env="azureml://registries/azureml/environments/sklearn-1.5-ubuntu22.04-py39-cpu/versions/1"
# 3. Deploy using the already-registered model
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name= endoint_name, # "diabetes-endpoint",
    model=registered_model,              # ← pulled from MyWorkspace1
    code_configuration=CodeConfiguration(
        code="./",
        scoring_script="score.py"
    ),
    environment=env,
    instance_type="Standard_F4s_v2",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment).result()
print("Deployment done!")
print(f"Done! Endpoint: {endpoint_name}")

# 4. Route all traffic to this deployment
endpoint.traffic = {"blue": 100}

try:
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
except Exception as e:
    print(type(e))
    print(str(e))   # full error text
    raise
#ml_client.online_endpoints.begin_create_or_update(endpoint).result()

final_endpoint=ml_client.online_endpoints.get(name=endpoint_name)
endpoint_keys=ml.client.online_endpoints.get_keys(name=endpoint_name)

print("\n-- Deployment Successful --")
print(f"Secure HTPPS Gateway URI: {final_endpoint.scoring_uri}")
print(f"Primary API Token: {endpoint_keys.primary_key}")