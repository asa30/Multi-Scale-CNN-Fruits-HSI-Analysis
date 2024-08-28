from GAN.myGANs import HSDWCGAN
from hsi_datagen import HSI_datagen
# get path from .env file
import os
import json
from dotenv import dotenv_values
import pandas as pd
# import adam optimiser
from keras.optimizers import Adam

config = dotenv_values(".env")
data_dir = config["DATA_DIR"]
working_dir = config["WORKING_DIR"]
# show all the folders and files in the working directory in a tree structure
os.chdir(data_dir)
os.listdir(data_dir)

train_all_v2 = json.load(open("./annotations/train_only_labeled_v2.json"))
val_v2 = json.load(open("./annotations/val_v2.json"))
test_v2 = json.load(open("./annotations/test_v2.json"))

train_records_df = pd.DataFrame(train_all_v2['records'])
# files attribute is a dictionary with keys: header_file, and data_file, convert these into seperate columns
train_records_df['header_file'] = train_records_df['files'].apply(lambda x: x['header_file'])
train_records_df['data_file'] = train_records_df['files'].apply(lambda x: x['data_file'])
train_records_df.drop('files', axis=1, inplace=True)

val_records_df = pd.DataFrame(val_v2['records'])
val_records_df['header_file'] = val_records_df['files'].apply(lambda x: x['header_file'])
val_records_df['data_file'] = val_records_df['files'].apply(lambda x: x['data_file']) 
val_records_df.drop('files', axis=1, inplace=True)

test_records_df = pd.DataFrame(test_v2['records'])
test_records_df['header_file'] = test_records_df['files'].apply(lambda x: x['header_file'])
test_records_df['data_file'] = test_records_df['files'].apply(lambda x: x['data_file'])
test_records_df.drop('files', axis=1, inplace=True)

train_annotations_df = pd.DataFrame(train_all_v2['annotations'])
train_df = pd.merge(train_records_df, train_annotations_df, on='id')

val_annotations_df = pd.DataFrame(val_v2['annotations'])
val_df = pd.merge(val_records_df, val_annotations_df, on='id')

test_annotations_df = pd.DataFrame(test_v2['annotations'])
test_df = pd.merge(test_records_df, test_annotations_df, on='id')

# merge all the dataframes
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

all_kiwi_NIR = all_df[(all_df['camera_type'] == 'NIR') & (all_df['fruit'] == 'Kiwi')]

# ripeness state mapping
ripeness_state_dict = {'unripe': 0, 'perfect': 1, 'overripe': 2}

# add ripeness state as a numerical value
all_kiwi_NIR['ripeness_state_y'] = all_kiwi_NIR['ripeness_state'].apply(lambda x: ripeness_state_dict[x])

# create datagen 
augConfig = {"keep_original": False, "horizontal_flip": True, "vertical_flip" : True, "noise":{"type":"poisson","level":0} }
all_datagen = HSI_datagen(all_kiwi_NIR, 'header_file', 'data_file', {'name': 'ripeness_state_y', 'type': int}, batch_size=9, target_size=(64, 64), data_dir=data_dir,
                            shuffle=True, normalize=False, augment= False, augmentConfig=augConfig, balance=True)

# creat adam optimiser
adam = Adam(lr=0.0002, beta_1=0.5)

#  create and run gan 
gan = HSDWCGAN(x=64, y=64, c=252, num_classes=3, ld=100, optimiser=adam, loss=['binary_crossentropy'], metrics=['accuracy'], use_HS_CNN=True, working_dir=working_dir + "gan_model1.1/")

# 
gan.train(epochs=10000, batch_size=9, sample_interval=50, save_interval=50, save_dir=working_dir + "gan_model1.1/", datagen=all_datagen)