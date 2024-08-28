import os
import sys
from dotenv import dotenv_values
config = dotenv_values(".env")
data_dir = config["DATA_DIR"]
working_dir = config["WORKING_DIR"]
sys.path.append(working_dir)
import pandas as pd 
import numpy as np 
from hsi_datagen import HSI_datagen
from sklearn.svm import SVC 

train_df = pd.read_csv(os.path.join(working_dir, "train_df.csv"))
val_df = pd.read_csv(os.path.join(working_dir, "val_df.csv"))
test_df = pd.read_csv(os.path.join(working_dir, "test_df.csv"))

#  get all rows where the camera type is VIS and the fruit is Kiwi
train_kiwi_VIS = train_df[(train_df['camera_type'] == 'VIS') & (train_df['fruit'] == 'Kiwi')]
# apply the same to the validation and test dataframes
val_kiwi_VIS = val_df[(val_df['camera_type'] == 'VIS') & (val_df['fruit'] == 'Kiwi')]
test_kiwi_VIS = test_df[(test_df['camera_type'] == 'VIS') & (test_df['fruit'] == 'Kiwi')]
train_kiwi_VIS = pd.concat([train_kiwi_VIS, val_kiwi_VIS])

# create data generators 
train_datagen = HSI_datagen(train_kiwi_VIS, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=len(train_kiwi_VIS), target_size=(64, 64), data_dir=data_dir,
                            shuffle=True, normalize=False, convert_to_tensor=False)
val_datagen = HSI_datagen(val_kiwi_VIS, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=len(val_kiwi_VIS), target_size=(64, 64), data_dir=data_dir,
                            shuffle=True, normalize=False, convert_to_tensor=False)
test_datagen = HSI_datagen(test_kiwi_VIS, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=len(test_kiwi_VIS), target_size=(64, 64), data_dir=data_dir,
                            shuffle=True, normalize=False, convert_to_tensor=False)

def flatten_data(X):
    return [x.flatten() for x in X]    

# train svc
svc = SVC()

for i in range(0,len(train_datagen)):
    X, y = train_datagen[i]
    svc.fit(flatten_data(X), y)

print("done training")

# evaluate svc
all_pred = []
all_true = []
for i in range(0,len(test_datagen)):
    X, y = test_datagen[i]
    y_pred = svc.predict(flatten_data(X))
    all_pred.append(y_pred)
    all_true.append(y)

accuracy = np.mean(np.array(all_pred) == np.array(all_true))
print(f"Accuracy: {accuracy}")