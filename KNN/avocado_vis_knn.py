import os
import sys
from dotenv import dotenv_values
config = dotenv_values("dmog.env")
data_dir = config["DATA_DIR"]
working_dir = config["WORKING_DIR"]
sys.path.append(working_dir)
import pandas as pd 
import numpy as np 
from hsi_datagen import HSI_datagen
from sklearn.neighbors import KNeighborsClassifier

train_df = pd.read_csv(os.path.join(working_dir, "train_df.csv"))
val_df = pd.read_csv(os.path.join(working_dir, "val_df.csv"))
test_df = pd.read_csv(os.path.join(working_dir, "test_df.csv"))

#  get all rows where the camera type is VIS and the fruit is Avocado
train_avocado_VIS = train_df[(train_df['camera_type'] == 'VIS') & (train_df['fruit'] == 'Avocado')]
# apply the same to the validation and test dataframes
val_avocado_VIS = val_df[(val_df['camera_type'] == 'VIS') & (val_df['fruit'] == 'Avocado')]
test_avocado_VIS = test_df[(test_df['camera_type'] == 'VIS') & (test_df['fruit'] == 'Avocado')]
train_avocado_VIS = pd.concat([train_avocado_VIS, val_avocado_VIS])

# create data generators 
train_datagen = HSI_datagen(train_avocado_VIS, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=len(train_avocado_VIS), target_size=(64, 64), data_dir=data_dir,
                            shuffle=True, normalize=False, convert_to_tensor=False)
val_datagen = HSI_datagen(val_avocado_VIS, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=len(val_avocado_VIS), target_size=(64, 64), data_dir=data_dir,
                            shuffle=True, normalize=False, convert_to_tensor=False)
test_datagen = HSI_datagen(test_avocado_VIS, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=len(test_avocado_VIS), target_size=(64, 64), data_dir=data_dir,
                            shuffle=True, normalize=False, convert_to_tensor=False)

def flatten_data(X):
    return [x.flatten() for x in X]    

knn = KNeighborsClassifier(n_neighbors=10, weights="distance", algorithm="ball_tree")

for i in range(0,len(train_datagen)):
    X, y = train_datagen[i]
    knn.fit(flatten_data(X), y)

print("done training")

# evaluate 
all_pred = []
all_true = []
for i in range(0,len(test_datagen)):
    X, y = test_datagen[i]
    y_pred = knn.predict(flatten_data(X))
    all_pred.append(y_pred)
    all_true.append(y)

accuracy = np.mean(np.array(all_pred) == np.array(all_true))
print(f"Accuracy: {accuracy}")