# get the key
resource_group=sina
account_name=godzillasinastorage
container_name=datasets

zip_filename=cats_and_dogs_filtered.zip
local_download_dir=~/cloudfiles/data/
data_folder=cats_and_dogs_filtered
full_directory_name=$local_download_dir$data_folder
echo $full_directory_name


wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip  -P $local_download_dir

unzip  $local_download_dir$zip_filename -d $local_download_dir

chmod 777 $full_directory_name



account_key=$(az storage account keys list -g $resource_group -n $account_name -o tsv --query "[0].{Value:value}")
az storage container create --name $container_name --auth-mode key --account-key $account_key --account-name $account_name -g $resource_group
az storage blob directory upload -c $container_name --auth-mode key --account-key $account_key --account-name $account_name -s $full_directory_name -d . --recursive

