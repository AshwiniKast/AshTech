# get-started/run-hello.py
import os

from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from dotenv import load_dotenv
load_dotenv()

auto_subscription = os.environ.get("AZURE_SUB_ID", "")
auto_resource_group = os.environ.get("CI_RESOURCE_GROUP", "AzureML1")
auto_workspace = os.environ.get("AZURE_WRKSPC_NAME", "MyWorkspace1")
ws = Workspace.from_config()
rg_name = os.environ.get("CI_RESOURCE_GROUP")
if not rg_name:
    raise ValueError("Critical Error: AZURE_RESOURCE_GROUP environment variable is not set!")

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

from azure.identity import DefaultAzureCredential

# If running locally for the first time, DefaultAzureCredential might pop open 
# a browser window asking you to log into your Azure account.
credential = DefaultAzureCredential()

try:
    # Looks for config.json automatically
    ml_client = MLClient.from_config(credential=credential)
    print(f"Successfully authenticated to {ml_client.workspace_name} via config.json!")
except Exception as e:
    print(f"Could not find config.json. Error: {e}")
# 1. Connect to your workspace
# ml_client = MLClient(
#     credential=DefaultAzureCredential(),
#     subscription_id=auto_subscription,
#     resource_group= rg_name, #auto_resource_group,
#     workspace_name= auto_workspace
# )

from azure.ai.ml import load_environment

# Gets the absolute path of the directory where this script lives
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Point directly to conda.yaml in that same folder
conda_path = os.path.join(current_dir, "conda.yaml")

# 2. Define the Environment metadata
pipeline_job_env = Environment(
    name="diabetes-training-env-v7",
   # version="1", #diabetes-training-env-v6",
    description="Custom environment for diabetes model training using SDK v2",
    tags={"scikit-learn": "latest", "mlflow": "latest","version": "v7"},
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04", # Standard Azure ML Base Image
    conda_file="conda.yaml"                                    # Path to t
)

# 3. Create/Register it in the cloud workspace
registered_env = ml_client.environments.create_or_update(pipeline_job_env)

print(f"Environment created with name: {registered_env.name} and version: {registered_env.version}")

from azure.ai.ml import command, Input
from azure.ai.ml.constants import AssetTypes
job = command(
    inputs={
    "data": Input(
        type=AssetTypes.URI_FOLDER,  # ← changed from URI_FILE
        path="azureml:DiaData1:1"
    )
    },
    code="./src", 
    command="python Predict_Diabetes.py --data ${{inputs.data}}",
    # Reference your registered environment here:
    #environment= "diabetes-training-env-v6@1", 
    environment=f"{registered_env.name}:{registered_env.version}",
    compute="ashkast991",
    experiment_name="day1-experiment-diabetes"
)

# Submit the job
ml_client.jobs.create_or_update(job)
# if __name__ == "__main__":
#     main()
# ws = Workspace.from_config()
# print(ws)
# experiment = Experiment(workspace=ws, name='day1-experiment-diabetes')

# config = ScriptRunConfig(source_directory='./src', script='Predict_Diabetes.py', compute_target='ashkast991')

# run = experiment.submit(config)
# aml_url = run.get_portal_url()
# print(aml_url)
