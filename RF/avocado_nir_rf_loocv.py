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
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv(os.path.join(working_dir, "train_df.csv"))
val_df = pd.read_csv(os.path.join(working_dir, "val_df.csv"))
test_df = pd.read_csv(os.path.join(working_dir, "test_df.csv"))

#  get all rows where the camera type is NIR and the fruit is Avocado
train_avocado_NIR = train_df[(train_df['camera_type'] == 'NIR') & (train_df['fruit'] == 'Avocado')]
# apply the same to the validation and test dataframes
val_avocado_NIR = val_df[(val_df['camera_type'] == 'NIR') & (val_df['fruit'] == 'Avocado')]
test_avocado_NIR = test_df[(test_df['camera_type'] == 'NIR') & (test_df['fruit'] == 'Avocado')]
all_avocado_NIR = pd.concat([train_avocado_NIR, val_avocado_NIR, test_avocado_NIR])


def flatten_data(X):
    return [x.flatten() for x in X]


# define Leave one out corss validation method 
def LOOCV(dataset, augment = False, augConfig = None, balance = False, verbose=0):
    # initialise empty list to store the results
    results = []
    all_pred = []
    # iterate through the dataset using iterrows
    i=0
    for index, row in dataset.iterrows():
        # reset the model weights
        rf = RandomForestClassifier(criterion='entropy', max_features='sqrt')
        # omit the current row from the dataset
        y = row['ripeness_state_y']
        temp_df = dataset.drop(index)
        # get the omitted row as a dataframe
        omitted_df = dataset.iloc[[i]]
        # get the data generator
        temp_datagen = HSI_datagen(temp_df, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=len(all_avocado_NIR)-1, target_size=(64, 64), data_dir=data_dir,
                                shuffle=True, normalize=False, augment=augment, augmentConfig=augConfig, balance=balance, convert_to_tensor=False)
        omitted_datagen = HSI_datagen(omitted_df, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=1, target_size=(64, 64), data_dir=data_dir,
                                shuffle=True, normalize=False, convert_to_tensor=False)
        # train the model
        print(f"Training model for iteration {i}")
        tX, ty = temp_datagen[0]
        rf.fit(flatten_data(tX), ty)
        print("done training")
        # evaluate the model
        x, _ = omitted_datagen.__getitem__(0)
        y_pred = rf.predict(flatten_data(x))
        all_pred.append(y_pred)
        # compare the predicted value with the actual value
        if y_pred == y:
            if verbose:
                print(f"Correct prediction for iteration {i}")
            results.append(1)
        else:
            if verbose:
                print(f"Incorrect prediction for iteration {i}")
            results.append(0)
        # increment the iteration number
        i+=1
    # return the results
    return results, all_pred

results, all_pred = LOOCV(all_avocado_NIR, augment=False, balance=False, verbose=1)

accuracy = np.mean(results)

print(f"Accuracy: {accuracy}")
np.save(os.path.join(working_dir, "RF/avocado_nir_rf_loocv.npy"), results)
np.save(os.path.join(working_dir, "RF/avocado_nir_rf_loocv_preds.npy"), all_pred)