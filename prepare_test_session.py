import os
import requests
import zipfile
import pandas as pd

data_folder = '.data'
csv_folder = 'csv'

url_to_download = 'https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_1/flair_1_toy_dataset.zip'  
download_file_name = 'dataset.zip'  #


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded and saved as '{save_path}'.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"File unzipped to '{extract_to}'.")

# folders
create_folder_if_not_exists(data_folder)
create_folder_if_not_exists(csv_folder)

# download toy-dataset
download_file(url_to_download, download_file_name)

# unzip dataset
unzip_file(download_file_name, data_folder)


# create csv files for toy-dataset

def list_all_files(dir: str, ext: str='.tif') -> list:
    files_list = []

    for root, _, files in os.walk(dir):
        for file in files:
            file_name = os.path.join(root, file)

            if file_name[-4:] == ext:
                files_list.append(file_name)

    return files_list


TRAIN_FILES = list_all_files('.data/flair_1_toy_aerial_train')
X_train = [x for x in TRAIN_FILES if "aerial" in x]
y_train = [x.replace('aerial', 'labels').replace('IMG', 'MSK').replace('img', 'msk') for x in X_train]
print("Train files indexed")


TEST_FILES = list_all_files('.data/flair_1_toy_aerial_test')
X_test = [x for x in TEST_FILES if "aerial" in x]
y_test = [x.replace('aerial', 'labels').replace('IMG', 'MSK').replace('img', 'msk') for x in X_test]
print("Test files indexed")

train_val_split = int(len(X_train) * 0.9)


train_df = pd.DataFrame({'IMG': X_train[:train_val_split], 'MSK': y_train[:train_val_split]})
val_df = pd.DataFrame({'IMG': X_train[train_val_split:], 'MSK': y_train[train_val_split:]})
test_df = pd.DataFrame({'IMG': X_test, 'MSK': y_test})

train_df.to_csv('csv/flair_train.csv', index=False)
val_df.to_csv('csv/flair_val.csv', index=False)
test_df.to_csv('csv/flair_test.csv', index=False)
print("CSV files created")