from all_models import *
import tensorflow as tf
import pandas as pd
import numpy as np 
keras = tf.keras
from keras.models import Sequential
from keras.layers import DepthwiseConv2D, AveragePooling2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Conv2D, SeparableConvolution2D, Activation, Input, Add, Lambda, concatenate, Attention
from keras.activations import sigmoid
from keras.backend import resize_images
from keras_adabound import AdaBound
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from hsi_datagen import HSI_datagen
from IPython.display import Image
# import model checkpoint
from keras.callbacks import ModelCheckpoint
from dotenv import dotenv_values
import os 
import pyswarms as ps
# check gpu availability
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# clear previos session 
# tf.keras.backend.clear_session()
if __name__ == '__main__':
    config = dotenv_values("dmog.env")
    data_dir = config["DATA_DIR"]
    working_dir = config["WORKING_DIR"]

    train_df = pd.read_csv(os.path.join(working_dir, "train_df.csv"))
    val_df = pd.read_csv(os.path.join(working_dir, "val_df.csv"))
    test_df = pd.read_csv(os.path.join(working_dir, "test_df.csv"))
    all_df = pd.concat([train_df, val_df, test_df])

    target_optimiser = RMSprop(learning_rate=0.001)
    target_loss = 'sparse_categorical_crossentropy' 
    target_size = (64, 64)
    target_size_3d = (64, 64, 252)
    b = 18
    e = 40
    augConfig = {"keep_original": False, "horizontal_flip": False, "vertical_flip" : True,"rotation_range": 90, "noise":{"type":"poisson","level":0} }
    model = build_MKDCNN_params
    fruit = "Avocado"
    camera = "NIR"

    fc_df = all_df[(all_df["fruit"] == fruit) & (all_df["camera_type"] == camera)]
    fc_datagen = HSI_datagen(fc_df, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=b, target_size=target_size, data_dir=data_dir,
                                shuffle=True, normalize=False, augment=True, augmentConfig=augConfig)

    def optimize_hyperparameters(params):
        losses = []
        print(params)
        for param in params:
            int_param = np.round(param).astype(int)  
            print(int_param)
            current_model = model(int_param, target_size_3d)
            current_model.compile(optimizer=target_optimiser, loss=target_loss)
            history = current_model.fit(fc_datagen, epochs=e)
            losses.append(history.history['loss'][-1])
        return losses

    # set cognitive and social parameters for PSO
    options = {'c1': 0.5, 'c2': 0.7, 'w':0.9}
    # set bounds for the hyperparameters
    lower_bound = np.array([1, 25, 4, 25, 7, 25])
    upper_bound = np.array([3, 35, 6, 35, 9, 35])
    bounds = (lower_bound, upper_bound)
    dim = 6 
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=dim, options=options, bounds=bounds)

    # Perform optimization
    cost, pos = optimizer.optimize(optimize_hyperparameters, iters=50)

    # Print the best hyperparameters and the corresponding accuracy
    best_learning_rate, best_hidden_layers = pos
    print(f'Best parameters: {pos}')
    print(f'Best loss achieved: {cost}')