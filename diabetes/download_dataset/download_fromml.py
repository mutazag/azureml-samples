# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset


import os
print(os.getcwd())
workspace = Workspace.from_config('diabetes/aml_config/config.json')

dataset = Dataset.get_by_id(workspace, '767ba047-d134-45bc-94e6-bdc6c0ee4f4b')
dataset.download(target_path='./diabetes/download_dataset/data', overwrite=False)