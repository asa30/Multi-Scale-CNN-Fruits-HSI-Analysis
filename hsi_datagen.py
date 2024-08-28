import tensorflow as tf
from tensorflow import keras
import cv2 
import numpy as np
import spectral.io.envi as envi

from keras.preprocessing.image import ImageDataGenerator

# create a custom image data generator, implementing rasterio to read the image data and return the image data as a numpy array
# source code: https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
class HSI_datagen(keras.utils.Sequence): 
    def __init__(
            self,
            df,
            hdr_col, 
            dat_col,
            y_col,
            batch_size = 1,
            target_size = None,
            data_dir = "./", 
            shuffle=True,
            normalize=True,
            augment=False,
            balance=False,
            augmentConfig=None, 
            seed = None,
            convert_to_tensor = True,
            verbose = 0
        ):
        self.df = df.copy().sample(frac=1).reset_index(drop=True)
        self.hdr_col = hdr_col
        self.dat_col = dat_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.target_size = target_size
        self.working_dir = data_dir
        self.shuffle = shuffle
        self.normalize = normalize
        self.augment = augment
        self.augmentConfig = augmentConfig
        self.balance = balance
        self.n = len(self.df)
        self.y_name = y_col['name']
        self.unique = np.sort(df[self.y_name].unique())
        self.n_classes = df[self.y_name].nunique()
        self.y_type = y_col['type']
        self.seed = seed
        self.verbose = 0
        self.convert_to_tensor = convert_to_tensor
        if self.seed is not None:
            np.random.seed(self.seed)

    def load_HSI(self, hdr, dat):
        return envi.open(hdr, dat)

    def resize_HSI(self, img, target_size):
        # reverse target size tuple to match cv2 convention
        target_size = target_size[::-1]
        # hsi_image is a 3D numpy array of shape (height, width, bands)
        resized_bands = []
        for band in range(img.shape[2]):
            # Extract the band
            band_image = img[:, :, band]
            # Resize the band
            resized_band = cv2.resize(band_image, target_size, interpolation=cv2.INTER_LINEAR)
            # Add the resized band to the list
            resized_bands.append(resized_band)
        # Stack the resized bands along the third axis
        resized_hsi_image = np.stack(resized_bands, axis=2)
        return resized_hsi_image
    
    def zoom_HSI(self, img, zoom_factor):
        # zoomed_bands = []
        # for band in range(img.shape[2]):
        #     band_image = img[:, :, band]
        #     # Zoom the band using cv2
        #     zoomed_band = cv2.resize(band_image, (0, 0), fx=zoom_factor[0], fy=zoom_factor[1], interpolation=cv2.INTER_LINEAR)
        #     # Add the zoomed band to the list
        #     zoomed_bands.append(zoomed_band)
        # # Stack the zoomed bands along the third axis
        # zoomed_hsi_image = np.stack(zoomed_bands, axis=2)
        # return zoomed_hsi_image
        return img
    
    def rotate_HSI(self, img, angle):
        rotated_bands = []
        for band in range(img.shape[2]):
            band_image = img[:, :, band]
            # Rotate the band using cv2 
            rotated_band = cv2.warpAffine(band_image, cv2.getRotationMatrix2D((band_image.shape[1] / 2, band_image.shape[0] / 2), angle, 1), (band_image.shape[1], band_image.shape[0]))
            # Add the rotated band to the list
            rotated_bands.append(rotated_band)
        # Stack the rotated bands along the third axis
        rotated_hsi_image = np.stack(rotated_bands, axis=2)
        return rotated_hsi_image

    def flip_HSI(self, img, direction):
        flipped_bands = []
        for band in range(img.shape[2]):
            band_image = img[:, :, band]
            if direction == 'horizontal':
                flipped_bands.append(cv2.flip(band_image, 1))
            elif direction == 'vertical':
                flipped_bands.append(cv2.flip(band_image, 0))
        flipped_image = np.stack(flipped_bands, axis=2)
        return flipped_image
    
    def add_noise_HSI(self, img, noise_type, noise_level):
        noisy_bands = []
        for band in range(img.shape[2]):
            band_image = img[:, :, band]
            if noise_type == 'gaussian':
                # Generate Gaussian noise
                noise = np.random.normal(0, noise_level, band_image.shape)
                # Add the noise to the band
                noisy_band = band_image + noise
            elif noise_type == 'poisson':
                # Generate Poisson noise
                noise = np.random.poisson(band_image)
                # Add the noise to the band
                noisy_band = band_image + noise
            # Clip the band to ensure the pixel values are within the valid range
            noisy_band = np.clip(noisy_band, 0, 1)
            # Add the noisy band to the list
            noisy_bands.append(noisy_band)
        # Stack the noisy bands along the third axis
        noisy_hsi_image = np.stack(noisy_bands, axis=2)
        return noisy_hsi_image

    # def random_cut_HSI(self, img, cut_size):
    #     return img

    # inspired by get_random_transform from keras.preprocessing.image.ImageDataGenerator
    def random_augment_HSI(self):
        operations = list(self.augmentConfig.keys())

        if "rotation_range" in operations:
            theta = np.random.uniform(-self.augmentConfig['rotation_range'], self.augmentConfig['rotation_range'])
        else:
            theta = 0
        
        if "zoom_range" in operations:
            zx, zy = np.random.uniform(1 - self.augmentConfig['zoom_range'], 1 + self.augmentConfig['zoom_range'], 2)
        else:
            zx, zy = 1, 1

        if "horizontal_flip" in operations:
            flip_horizontal = np.random.random() < 0.5 if self.augmentConfig['horizontal_flip'] else False
        else:
            flip_horizontal = False
        
        if "vertical_flip" in operations:
            flip_vertical = np.random.random() < 0.5 if self.augmentConfig['vertical_flip'] else False
        else:
            flip_vertical = False
        
        if "noise" in operations:
            noise_type = self.augmentConfig['noise']['type']
            noise_level = self.augmentConfig['noise']['level']
        else:
            noise_type = None
            noise_level = 0

        transform_parameters = {
            "theta": theta,
            # "tx": tx,
            # "ty": ty,
            # "shear": shear,
            "zx": zx,
            "zy": zy,
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical,
            "noise_type": noise_type,
            "noise_level": noise_level,
            # "channel_shift_intensity": channel_shift_intensity,
            # "brightness": brightness,
        }

        return transform_parameters

    # inspired by apply_augment from keras.preprocessing.image.ImageDataGenerator
    def apply_augment_HSI(self, img, transform_parameters):
        # apply rotation
        # aug_img = img.copy()
        if transform_parameters['theta'] != 0:
            img = self.rotate_HSI(img, transform_parameters['theta'])
        # apply zoom
        if transform_parameters['zx'] != 1 or transform_parameters['zy'] != 1:
            img = self.zoom_HSI(img, (transform_parameters['zx'], transform_parameters['zy']))
        # apply horizontal flip
        if transform_parameters['flip_horizontal']:
            img = self.flip_HSI(img, 'horizontal')
        # apply vertical flip
        if transform_parameters['flip_vertical']:
            img = self.flip_HSI(img, 'vertical')
        # apply noise
        if transform_parameters['noise_type'] is not None and transform_parameters['noise_level'] > 0:
            img = self.add_noise_HSI(img, transform_parameters['noise_type'], transform_parameters['noise_level'])
        return img

    def augment_HSI(self, X, y):
        newX = []
        newY = []

        # get keylist from augmentConfig
        # check if augmentConfig contains the required parameters
        # if not, use default values

        operations = list(self.augmentConfig.keys())

        for img, label in zip(X, y): 
            if 'keep_original' in operations and self.augmentConfig['keep_original'] == True:
                newX.append(img)
                newY.append(label)
            # apply number of augmentations 
            if 'augmentations' in operations:
                for i in range(self.augmentConfig['augmentations']):
                    aug = self.random_augment_HSI()
                    new_img = self.apply_augment_HSI(img, aug)
                    newX.append(new_img)
                    newY.append(label)
            else:
                aug = self.random_augment_HSI()
                new_img = self.apply_augment_HSI(img, aug)
                newX.append(new_img)
                newY.append(label)
        return newX, newY
    
    """
    Balance the dataset by augmenting the minority classes and also undersampling the majority classes based on the batch size
    """
    def balance_HSI(self, X, y):
        # get the unique classes
        unique = self.unique
        if self.verbose:
            print(f"unique classes: {unique}")
        # get the maximum count
        max_count = self.batch_size//len(unique)
        # construct a dictionaty to store the indices of each class
        indices = {i : np.where(y == i)[0] for i in unique}
        # create a list to store the balanced data
        balanced_X = []
        balanced_y = []
        # for each unique class
        for i in unique:
            curr_class = i
            # get the number of samples in the class
            n_samples = len(indices[i])
            samples = indices[i]
            # first optimistic case: if the number of samples is equal to the maximum count
            if n_samples == max_count:
                if self.verbose:
                    print(f"class {curr_class} is already balanced")
                for j in samples: 
                    balanced_X.append(X[j])
                    balanced_y.append(curr_class)
            # second optimistic case: if the number of samples is greater than the maximum count
            elif n_samples > max_count:
                if self.verbose:
                    print(f"class {curr_class} was over the max count of {max_count} samples, {n_samples} samples were found")
                for j in np.random.choice(samples, max_count, replace=False):
                    balanced_X.append(X[j])
                    balanced_y.append(curr_class)
            # worst case: if the number of samples is zero
            elif n_samples == 0:
                if self.verbose:
                    print(f"no samples were found for class {curr_class}")
                n_add = max_count
                # retrieve some examples from the dataframe and add them to the balanced data
                filtered = self.df[self.df[self.y_name] == curr_class]
                for j in range(n_add):
                    idx = np.random.choice(filtered.index, 1)[0]
                    new_img = self.__get_input((self.working_dir + filtered.loc[idx][self.hdr_col], self.working_dir + filtered.loc[idx][self.dat_col]))
                    if self.augment:
                        aug = self.random_augment_HSI()
                        new_img = self.apply_augment_HSI(new_img, aug)
                    balanced_X.append(new_img)
                    balanced_y.append(curr_class)
            # first pessimistic case: if the number of samples is less than the maximum count 
            elif n_samples < max_count:
                if self.verbose:
                    print(f"class {curr_class} was under the max count of {max_count} samples, {n_samples} samples were found")
                # first add the already existing samples
                for j in samples: 
                    balanced_X.append(X[j])
                    balanced_y.append(curr_class)
                # get the number of samples to be added
                n_add = max_count - n_samples
                # augment the already existing samples to reach the maximum count
                for j in range(n_add):
                    aug = self.random_augment_HSI()
                    random_index = np.random.choice(samples, 1)[0]
                    new_img = self.apply_augment_HSI(X[random_index], aug)
                    balanced_X.append(new_img)
                    balanced_y.append(curr_class)
        # return the balanced data
        return balanced_X, balanced_y

    def __len__(self):
        return self.n // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path, bbox=None):
        # open the image file
        # read the image data
        img_data = self.load_HSI(path[0], path[1])
        # convert the image data to a numpy array
        # img_data = np.array([img_data[:,:,i] for i in range(img_data.shape[2])])
        # crop the image data to the bounding box
        if bbox is not None:
            img_data = img_data[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
        # resize the image data to the target size
        if self.target_size is not None:
            # changed
            img_data = self.resize_HSI(img_data, self.target_size)
        # normalize the image data
        if self.normalize:
            pass
        return img_data
    
    def __get_output(self, labels, num_classes):
        # convert the labels into onehot encoded vectors based on the type of label
        # for now return the labels as is
        return labels
    
    def __get_data(self, batch):
        # get the input data
        X = np.array([self.__get_input((self.working_dir + batch.iloc[i][self.hdr_col], self.working_dir + batch.iloc[i][self.dat_col])) for i in range(len(batch))])
        # get the output data
        y = self.__get_output(batch[self.y_name], self.n_classes)
        # y=None
        if self.augment:
            X,y = self.augment_HSI(X, y)
        if self.balance:
            X, y = self.balance_HSI(X, y)
        return X, y

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch = self.df[start:end]
        X, y = self.__get_data(batch)
        # self.on_epoch_end()
        # convert to tensor
        if self.convert_to_tensor:
            return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32)
        else:
            return X, y    