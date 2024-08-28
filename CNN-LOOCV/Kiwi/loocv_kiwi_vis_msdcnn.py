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
from all_models import build_MSDCNN
from keras_adabound import AdaBound
from keras.optimizers import Adam, RMSprop

train_df = pd.read_csv(os.path.join(working_dir, "train_df.csv"))
val_df = pd.read_csv(os.path.join(working_dir, "val_df.csv"))
test_df = pd.read_csv(os.path.join(working_dir, "test_df.csv"))

#  get all rows where the camera type is VIS and the fruit is Kiwi
train_kiwi_VIS = train_df[(train_df['camera_type'] == 'VIS') & (train_df['fruit'] == 'Kiwi')]
# apply the same to the validation and test dataframes
val_kiwi_VIS = val_df[(val_df['camera_type'] == 'VIS') & (val_df['fruit'] == 'Kiwi')]
test_kiwi_VIS = test_df[(test_df['camera_type'] == 'VIS') & (test_df['fruit'] == 'Kiwi')]
all_kiwi_VIS = pd.concat([train_kiwi_VIS, val_kiwi_VIS, test_kiwi_VIS])

def LOOCV(model, dataset, batch_size, epochs=5, augment = False, augConfig = None, balance = False, save_path="", verbose=0):
    # initialise empty list to store the results
    results = []
    # save starting weights of the model
    if not os.path.exists(os.path.join(working_dir, save_path)):
            os.mkdir(os.path.join(working_dir, save_path))
    model.save_weights(os.path.join(working_dir, save_path + f"/{model.__name__}_weights_start.h5"))
    # iterate through the dataset using iterrows
    i=0
    for index, row in dataset.iterrows():
        # reset the model weights
        if verbose:
            print(f"Training model for iteration {i}")
            print("resetting model weights")
        model.load_weights(os.path.join(working_dir, save_path + f"/{model.__name__}_weights_start.h5"))
        # omit the current row from the dataset
        y = row['ripeness_state_y']
        temp_df = dataset.drop(index)
        # get the omitted row as a dataframe
        omitted_df = dataset.iloc[[i]]
        # get the data generator
        temp_datagen = HSI_datagen(temp_df, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=batch_size, target_size=(64, 64), data_dir=data_dir,
                                shuffle=True, normalize=False, augment=augment, augmentConfig=augConfig, balance=balance)
        omitted_datagen = HSI_datagen(omitted_df, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=1, target_size=(64, 64), data_dir=data_dir,
                                shuffle=True, normalize=False)
        # train the model
        model.fit(temp_datagen, epochs=epochs)
        # save the model weights keeping track of iteration number 
        # create a folder, if it does not exist 
        model.save_weights(os.path.join(working_dir, save_path + f"/{model.__name__}_weights_{i}.h5"))
        # evaluate the model
        x, _ = omitted_datagen.__getitem__(0)
        y_pred = model.predict(x)
        y_pred = np.argmax(y_pred, axis=1)
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
    return results

# build the model
kiwi_vis_model = build_MSDCNN((64, 64, 224))

kiwi_vis_model.compile(optimizer=RMSprop(learning_rate=1e-2), loss='sparse_categorical_crossentropy')

kiwi_vis_loocv_results = LOOCV(kiwi_vis_model, all_kiwi_VIS, 12, epochs=40, save_path="kiwi_vis_msdcnn_loocv", verbose=1)

print(f"Accuracy of LOOCV for MSDCNN model tested on Kiwi VIS: {np.mean(kiwi_vis_loocv_results)}")

# save the results
np.save(os.path.join(working_dir, "kiwi_vis_msdcnn_loocv_results.npy"), kiwi_vis_loocv_results)
