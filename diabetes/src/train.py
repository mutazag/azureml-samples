# %%
import numpy as np
# from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sys
import mlflow
import mlflow.sklearn
import pandas as pd


#%% 
# arguments=[
#         '--kernel', 'linear', 
#         '--penalty', 1.0,
#         '--train_dataset', input_dataset.as_named_input('train')],

import sys 
print('############################### sys args')
print(sys.argv)
import argparse
parser = argparse.ArgumentParser()

if (parser.prog == 'ipykernel_launcher.py'):
    parser.add_argument('--train_dataset', dest='train_dataset', default='../download_dataset/data-csv/diabetes.csv')
    parser.add_argument('--dataset_download', dest='dataset_download', default='../download_dataset/data-parquet')
    parser.add_argument('--input_dataset', dest='input_dataset', default='../download_dataset/data-parquet')
    
else:
    parser.add_argument('--train_dataset', dest='train_dataset', default='./diabetes/download_dataset/data-csv/diabetes.csv')
    parser.add_argument('--dataset_download', dest='dataset_download', default='./diabetes/download_dataset/data-parquet')
    parser.add_argument('--input_dataset', dest='input_dataset', default='./diabetes/download_dataset/data-parquet')


#%%
args, _ = parser.parse_known_args()
train_dataset = args.train_dataset
print('###############################')
print('train dataset : ', train_dataset)
print('dataset_download : ', args.dataset_download)

#%% 
mlflow.set_tag('train_dataset', train_dataset)
mlflow.set_tag('dataset_download', args.dataset_download)
mlflow.set_tag('input_dataset', args.input_dataset)
#%% 
import os
print('################################ download dir content')
print(os.listdir(args.dataset_download))
mlflow.log_dict({'files': os.listdir(args.dataset_download)}, 'downloadfiles.json')
mlflow.log_dict(dict(os.environ),'environ.json')
df_input = pd.read_parquet(args.dataset_download)
# df_input = pd.read_parquet(args.input_dataset)
print(df_input.head())


# %% 
try: 
    #The download location can also be retrieved from input_datasets of the run context.
    from azureml.core import Run
    download_location = Run.get_context().input_datasets['input_dataset']
    print('################################ download location')
    print(download_location)
    print(os.listdir(download_location))
    df2_input = pd.read_parquet(download_location)
    print(df2_input.tail())
except: 
    print('no input_dataset')
#%%

print('################################ read train dataset')
X = np.array(df_input.drop(columns=['target']))
y = np.array(df_input.target)
print('read features: ', len(X))
print('read target: ', len(y))
#%%
# diabetes = load_diabetes(as_frame=True)
# df = diabetes.data
# df['target'] = diabetes.target
# df.to_csv("diabetes.csv", index=False)
#%%
# df = pd.read_csv("diabetes.csv")
# X = np.array(df.drop(columns=['target']))
# y = np.array(df.target)
#%%
alpha = .5

#%%

columns = ["age", "gender", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}
#%%

mlflow.log_metric("Training samples", len(data["train"]["X"]))
mlflow.log_metric("Test samples", len(data["test"]["X"]))

# Log the algorithm parameter alpha to the run
mlflow.log_metric("alpha", alpha)
#%%
# Create, fit, and test the scikit-learn Ridge regression model
regression_model = Ridge(alpha=0.03)
regression_model.fit(data["train"]["X"], data["train"]["y"])
preds = regression_model.predict(data["test"]["X"])

# Log mean squared error
print("Mean Squared Error is", mean_squared_error(data["test"]["y"], preds))
mlflow.log_metric("mse", mean_squared_error(data["test"]["y"], preds))
#%%
# Save the model to the outputs directory for capture
mlflow.sklearn.log_model(regression_model, "diabetes_regression_model")

#%%
# Plot actuals vs predictions and save the plot within the run
import matplotlib.pyplot as plt
fig = plt.figure(1)
idx = np.argsort(data["test"]["y"])
plt.plot(data["test"]["y"][idx], preds[idx])
fig.savefig("actuals_vs_predictions.png")
mlflow.log_artifact("actuals_vs_predictions.png")
# %%

# %%
