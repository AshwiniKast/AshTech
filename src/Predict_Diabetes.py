#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import sys
import subprocess

# Force install azure-ai-ml if it's missing from this specific runtime environment
try:
    from azure.ai.ml import MLClient
except ImportError:
    print("azure.ai.ml not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "azure-ai-ml", "azure-identity","pandas", "numpy","azureml-fsspec","scikit-learn","matplotlib","azureml-mlflow","azureml-core"])
    from azure.ai.ml import MLClient
    print("Successfully installed and imported azure.ai.ml!")

# azureml-core of version 1.0.72 or higher is required
#from azureml.core import Workspace, Dataset

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.operations import DataOperations

# 1. Initialize the modern client (replaces Workspace)
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# 2. Fetch your data asset (replaces Dataset)
# Replace "your_dataset_name" with the actual name of your registered data
#data_asset = ml_client.data.get(name="DiaData1", version="1")

#print(f"Successfully connected to Data Asset: {data_asset.name}")
#print(f"Data URI path: {data_asset.path}")

# subscription_id = ''
# resource_group = 'AzureML1'
# workspace_name = 'MyWorkspace1'

# workspace = Workspace(subscription_id, resource_group, workspace_name)
# data_asset = ml_client.data.get(name="DiaData1", version="1")
# print(f"Data URI path: {data_asset.path}")
# # dataset = Dataset.get_by_name(workspace, name='DiaData1')
# #ml_client.data.download(name="DiaData1",download_path='./data', overwrite=False)
# path = data_asset.path
# if path.startswith("azureml://"):
#     # If it's an internal path, we fallback to pulling its direct storage path
#     # or you can read it directly using the underlying datastore info.
#     print(f"Asset path is: {path}")

# If your data asset was registered via a path that pandas can reach natively:
#df = pd.read_csv(data_asset.path)

# In[2]:

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
args = parser.parse_args()

import os

df = pd.read_csv(os.path.join(args.data, "diabetes.csv"))  # ← add your actual filename # ← Azure ML resolves the path automatically

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Initialize the v2 ML Client
#ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)


# Get the cloud storage URI directly
#print(data_asset.path)
# Output example: azureml://datastores/workspaceblobstore/paths/training-data/


# In[3]:


import pandas as pd
#df=pd.read_csv(path)
# df


# In[4]:


# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# %%
#df=pd.read_csv('/Users/ashwi/Documents/MyRepo1/diabetes.csv')

# %%
df.info()

# %%
#print("Nulls")
#print("======")
#print(df.isnull().sum())

# %%
#print("0s")
#print("===")
#print(df.eq(0).sum())

# %%
df[['Glucose','BloodPressure','skinthickness','Insulin','BMI','diabetespedigreefunction','Age']]=df[['glucose','bloodpressure','skinthickness','insulin','bmi','diabetespedigreefunction','age']].replace(0,np.nan)

# %%
df.fillna(df.mean(),inplace=True)
#print(df.eq(0).sum())

# %%
corr=df.corr()
#print(corr)

# %%
#matplotlib inline
import matplotlib.pyplot as plt

fig, ax=plt.subplots(figsize=(10,10))
cax=ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)
ticks=np.arange(0, len(df.columns),1)
ax.set_xticks(ticks)

ax.set_xticklabels(df.columns)
plt.xticks(rotation=90)

ax.set_yticklabels(df.columns)
plt.yticks(ticks)

#print the correlation factor

for i in range(df.shape[1]):
    for j in range(corr.shape[1]):
        value = corr.iat[i, j]
        text=ax.text(j, i, round(value,2), ha="center", va="center",color="w")
#plt.show()        

# %%
#Logistic Regression

from sklearn import linear_model
from sklearn.model_selection import cross_val_score

X=df[['Glucose','BMI','Age']]

y=df.iloc[:,8]

log_regress=linear_model.LogisticRegression()
log_Regress_score=cross_val_score(log_regress, X, y, cv=10, scoring='accuracy').mean()

#print(log_Regress_score)

result=[]
result.append(log_Regress_score)

# %%
# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
cv_scores=[]
folds=10

#Creating odd list for K for KNN
ks=list(range(1, int(len(X)*((folds-1)/folds)), 2))

#Perform k fold cross validation
for k in ks:
    knn=KNeighborsClassifier(n_neighbors=k)
    score=cross_val_score(knn, X, y, cv=folds, scoring='accuracy').mean()
    cv_scores.append(score)
    
    knn_score=max(cv_scores)
#Find optimal k that gives the highest score
optimal_k=ks[cv_scores.index(knn_score)]

#print(f"optimal no of neighbors is {optimal_k}")
print(knn_score)
result.append(knn_score)

# %%
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2,random_state=42 )
knn=KNeighborsClassifier(n_neighbors=19)
knn.fit(X_train, y_train)

# %%
import pickle
filename='diabetes1.sav'

pickle.dump(knn, open(filename, 'wb'))

# %%
import warnings
warnings.filterwarnings('ignore')
# loaded_model=pickle.load(open(filename, 'rb'))

# Glucose=65
# BMI=70
# Age=50
# prediction=loaded_model.predict([[Glucose, BMI, Age]])
# #print(prediction)
# if (prediction[0]==0):
#     #print("Non-diabetic")
# else:
#     #print("Diabetic")   

# %%
#proba=loaded_model.predict_proba([[Glucose,BMI,Age]])
#print(proba)
#print("confidence :" +str(round(np.amax(proba[0])* 100, 2))+ "%")

# %%





# In[15]:


#%pip install mlflow azureml-mlflow


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix, 
    ConfusionMatrixDisplay
)

# --- Assuming you already have: ---
# knn (your trained KNeighborsClassifier)
# X_test (test features)
# y_test (true test labels)

print("Evaluating KNN model performance...")

# 1. Generate Predictions
y_pred = knn.predict(X_test)

# Predict probabilities (required for ROC-AUC score)
# KNN supports this out of the box
y_proba = knn.predict_proba(X_test)[:, 1] 

# 2. Calculate Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-score: {f1:.4f}")
print(f"ROC_AUC: {roc_auc}")
# 3. Log Metrics directly to Azure ML via MLflow
# mlflow.log_metric("test_accuracy", accuracy)
# mlflow.log_metric("test_precision", precision)
# mlflow.log_metric("test_recall", recall)
# mlflow.log_metric("test_f1_score", f1)
# mlflow.log_metric("test_roc_auc", roc_auc)

# 4. Generate & Log a Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])

# Plot using matplotlib
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title("KNN Classifier - Confusion Matrix")

# Log the plot asset straight to Azure ML Studio
mlflow.log_figure(fig, "evaluation_plots/confusion_matrix.png")
plt.close(fig) # Clean up memory

print("Evaluation metrics and plots successfully logged to Azure ML!")


# In[6]:


import pickle

# Let's assume 'clf' is your trained model object (like a Scikit-Learn classifier)
model_filename = "./model.pkl"

# Save the model to your current working directory
with open(model_filename, "wb") as file:
    pickle.dump(knn, file)

print(f"Model successfully saved locally as {model_filename}")


# In[8]:


from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model_asset = Model(
    name="knn_diabetes_model",         # <--- Name your Azure Asset here
   # version="1",
    path="./model.pkl",                  # The actual pickled file
    type=AssetTypes.CUSTOM_MODEL,
    description="KNN model for predicting diabetes"
)
registered_model = ml_client.models.create_or_update(model_asset)

# In[9]:


#from azureml.core.model import Model
# model=Model.register(workspace=workspace,
#                 model_path="./model.pkl",
#                 model_name="knn_diabetes_model",
#                 tags={"version": "1.0"},
#                 description="KNN model to predict Diabetes")

