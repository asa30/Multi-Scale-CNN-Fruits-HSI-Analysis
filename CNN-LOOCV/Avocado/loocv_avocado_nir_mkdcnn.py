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
from all_models import build_MKDCNN
from keras_adabound import AdaBound
from keras.optimizers import Adam, RMSprop

train_df = pd.read_csv(os.path.join(working_dir, "train_df.csv"))
val_df = pd.read_csv(os.path.join(working_dir, "val_df.csv"))
test_df = pd.read_csv(os.path.join(working_dir, "test_df.csv"))

# get all rows where the camera type is NIR and the fruit is Avocado 
train_avocado_NIR = train_df[(train_df['camera_type'] == 'NIR') & (train_df['fruit'] == 'Avocado')]
# apply the same to val and test dataframes
val_avocado_NIR = val_df[(val_df['camera_type'] == 'NIR') & (val_df['fruit'] == 'Avocado')]
test_avocado_NIR = test_df[(test_df['camera_type'] == 'NIR') & (test_df['fruit'] == 'Avocado')]
all_avocado_NIR = pd.concat([train_avocado_NIR, val_avocado_NIR, test_avocado_NIR])

augConfig = {"keep_original": False, "horizontal_flip": False, "vertical_flip" : True,"rotation_range": 90, "noise":{"type":"poisson","level":0} }

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

avocado_nir_model = build_MKDCNN((64, 64, 252))

avocado_nir_model.compile(optimizer=RMSprop(learning_rate=1e-2), loss='sparse_categorical_crossentropy')

avocado_nir_loocv_results = LOOCV(avocado_nir_model, all_avocado_NIR, 12, 40, save_path="avocado_nir_mkdcnn_loocv", verbose=1)

print(f"Accuracy of LOOCV for MKDCNN model tested on Avocado NIR: {np.mean(avocado_nir_loocv_results)}")
# save the results
np.save(os.path.join(working_dir, "avocado_nir_mkdcnn_loocv_results.npy"), avocado_nir_loocv_results)