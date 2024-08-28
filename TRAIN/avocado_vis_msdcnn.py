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

#  get all rows where the camera type is VIS and the fruit is Avocado
train_avocado_VIS = train_df[(train_df['camera_type'] == 'VIS') & (train_df['fruit'] == 'Avocado')]
# apply the same to the validation and test dataframes
val_avocado_VIS = val_df[(val_df['camera_type'] == 'VIS') & (val_df['fruit'] == 'Avocado')]
test_avocado_VIS = test_df[(test_df['camera_type'] == 'VIS') & (test_df['fruit'] == 'Avocado')]
all_avocado_VIS = pd.concat([train_avocado_VIS, val_avocado_VIS, test_avocado_VIS])
augConfig = {"keep_original": False, "horizontal_flip": False, "vertical_flip" : True,"rotation_range": 90, "noise":{"type":"poisson","level":0} }
# create a data generator for the Avocado VIS data
train_avocado_VIS_datagen = HSI_datagen(all_avocado_VIS, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=9, target_size=(64, 64), data_dir=data_dir,
                            shuffle=True, normalize=False, augment= True, augmentConfig=augConfig, balance=True)

# build the model
model = build_MSDCNN((64, 64, 224))

# compile the model
model.compile(optimizer=RMSprop(learning_rate=1e-2), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_avocado_VIS_datagen, epochs=100)