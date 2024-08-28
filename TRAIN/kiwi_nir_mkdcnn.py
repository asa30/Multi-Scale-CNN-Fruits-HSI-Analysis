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

#  get all rows where the camera type is NIR and the fruit is Kiwi
train_kiwi_NIR = train_df[(train_df['camera_type'] == 'NIR') & (train_df['fruit'] == 'Kiwi')]
# apply the same to the validation and test dataframes
val_kiwi_NIR = val_df[(val_df['camera_type'] == 'NIR') & (val_df['fruit'] == 'Kiwi')]
test_kiwi_NIR = test_df[(test_df['camera_type'] == 'NIR') & (test_df['fruit'] == 'Kiwi')]
all_kiwi_NIR = pd.concat([train_kiwi_NIR, val_kiwi_NIR, test_kiwi_NIR])
augConfig = {"keep_original": False, "horizontal_flip": False, "vertical_flip" : True,"rotation_range": 90, "noise":{"type":"poisson","level":0} }
# create a data generator for the Kiwi NIR data
train_kiwi_NIR_datagen = HSI_datagen(all_kiwi_NIR, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=9, target_size=(64, 64), data_dir=data_dir,
                            shuffle=True, normalize=False, augment= True, augmentConfig=augConfig, balance=True)

# build the model
model = build_MKDCNN((64, 64, 252))

# compile the model
model.compile(optimizer=RMSprop(learning_rate=1e-2), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_kiwi_NIR_datagen, epochs=100)