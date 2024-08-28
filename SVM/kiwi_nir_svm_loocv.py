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

#  get all rows where the camera type is NIR and the fruit is Kiwi
train_kiwi_NIR = train_df[(train_df['camera_type'] == 'NIR') & (train_df['fruit'] == 'Kiwi')]
# apply the same to the validation and test dataframes
val_kiwi_NIR = val_df[(val_df['camera_type'] == 'NIR') & (val_df['fruit'] == 'Kiwi')]
test_kiwi_NIR = test_df[(test_df['camera_type'] == 'NIR') & (test_df['fruit'] == 'Kiwi')]
all_kiwi_NIR = pd.concat([train_kiwi_NIR, val_kiwi_NIR, test_kiwi_NIR])

# all_datagen = HSI_datagen(all_kiwi_NIR, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=len(all_kiwi_NIR), target_size=(64, 64), data_dir=data_dir,
                            # shuffle=True, normalize=False, convert_to_tensor=False)

def flatten_data(X):
    return [x.flatten() for x in X]

# def fit_svc(svc, datagen):
#     for i in range(0,len(datagen)):
#         X, y = datagen[i]
#         svc.fit(flatten_data(X), y)
#     return svc

# define Leave one out corss validation method 
def SVC_LOOCV(dataset, augment = False, augConfig = None, balance = False, verbose=0):
    # initialise empty list to store the results
    results = []
    all_pred = []
    # iterate through the dataset using iterrows
    i=0
    for index, row in dataset.iterrows():
        # reset the model weights
        svc = SVC(kernel='rbf')
        # omit the current row from the dataset
        y = row['ripeness_state_y']
        temp_df = dataset.drop(index)
        # get the omitted row as a dataframe
        omitted_df = dataset.iloc[[i]]
        # get the data generator
        temp_datagen = HSI_datagen(temp_df, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=len(all_kiwi_NIR)-1, target_size=(64, 64), data_dir=data_dir,
                                shuffle=True, normalize=False, augment=augment, augmentConfig=augConfig, balance=balance, convert_to_tensor=False)
        omitted_datagen = HSI_datagen(omitted_df, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=1, target_size=(64, 64), data_dir=data_dir,
                                shuffle=True, normalize=False, convert_to_tensor=False)
        # train the model
        print(f"Training model for iteration {i}")
        tX, ty = temp_datagen[0]
        svc.fit(flatten_data(tX), ty)
        print("done training")
        # evaluate the model
        x, _ = omitted_datagen.__getitem__(0)
        y_pred = svc.predict(flatten_data(x))
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

results, all_pred = SVC_LOOCV(all_kiwi_NIR, augment=False, balance=False, verbose=1)

accuracy = np.mean(results)

print(f"Accuracy: {accuracy}")
np.save(os.path.join(working_dir, "SVM/kiwi_nir_svm_loocv.npy"), results)
np.save(os.path.join(working_dir, "SVM/kiwi_nir_svm_loocv_preds.npy"), all_pred)