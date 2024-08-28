import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, BatchNormalization, Activation, Embedding, LeakyReLU
from keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D, ZeroPadding3D, UpSampling3D, Conv3D, Conv3DTranspose, MaxPooling3D
from keras.layers import DepthwiseConv2D, AveragePooling2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Conv2D, SeparableConvolution2D, Activation
from keras.models import Sequential, Model
from keras.activations import sigmoid
from keras_adabound import AdaBound
from keras.optimizers import Adam, SGD, Adamax
import numpy as np
import matplotlib.pyplot as plt
import datetime
from hsi_datagen import HSI_datagen

# the initial template for this conditional gan can be found at : 
# https://github.com/eriklindernoren/Keras-GAN

class HSDWCGAN():
    def __init__(self, x=None, y=None, c=None, num_classes=None, ld = 100, optimiser = None, loss = None, metrics = None, use_HS_CNN=False, working_dir="./"):
        self.rows = x
        self.cols = y
        self.channels = c
        self.img_shape = (x,y,c) 

        self.working_dir = working_dir
        
        self.num_classes = num_classes
        self.latent_dim = ld

        self.optimiser = optimiser
        self.loss = loss
        self.metrics = metrics

        self.use_HS_CNN = use_HS_CNN

        # build and compile the discriminator
        self.discriminator = self.build_discriminator(self.img_shape)
        self.discriminator.compile(loss=self.loss, optimizer=self.optimiser, metrics=self.metrics)

        # build the generator
        self.generator = self.build_generator()

        # the generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # for the combined model we will only train the generator
        self.discriminator.trainable = False

        # the discriminator takes generated image as input and determines validity
        valid = self.discriminator([img, label])

        # the combined model (stacked generator and discriminator)
        # trains the generator to fool the discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=self.loss, optimizer=self.optimiser)

    # def build_generator(self):
    #     # sequential model
    #     model = Sequential()

    #     xi = self.rows//4
    #     yi = self.cols//4

    #     model.add(Dense(128 * xi * yi, activation="relu", input_dim=self.latent_dim))
    #     model.add(Reshape((xi, yi, 128)))
    #     model.add(UpSampling2D())
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(DepthwiseConv2D(kernel_size=3, padding="same"))
    #     model.add(Conv2D(128, kernel_size=3, padding="same"))
    #     model.add(Activation("relu"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(UpSampling2D())
    #     model.add(DepthwiseConv2D(kernel_size=3, padding="same"))
    #     model.add(Conv2D(64, kernel_size=3, padding="same"))
    #     model.add(Activation("relu"))
    #     model.add(BatchNormalization(momentum=0.8))
    #     model.add(DepthwiseConv2D(kernel_size=3, padding="same"))
    #     model.add(Conv2D(self.channels, kernel_size=3, padding="same", activation="tanh"))
        
    #     model.summary()

    #     noise = Input(shape=(self.latent_dim,))
    #     label = Input(shape=(1,))
    #     label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

    #     model_input = multiply([noise, label_embedding])
    #     img = model(model_input)

    #     return Model([noise, label], img)
    
    def build_generator(self):
        # sequential model
        model = Sequential()

        xi = self.rows
        yi = self.cols

        # Increase the initial number of neurons to improve the generator's capacity.
        # Use LeakyReLU instead of ReLU for better gradient flow during training.
        model.add(Dense(128 * xi * yi, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((xi, yi, 128)))  # Adjust this according to the first Dense layer's adjustments.
        model.add(BatchNormalization())
        model.add(ReLU())

        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(ReLU())

        # Added more layers with downsampling to improve feature extraction.
        model.add(DepthwiseConv2D(kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(ReLU())

        model.add(DepthwiseConv2D(kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(512, kernel_size=3, padding="same"))
        model.add(ReLU())

        # Final layer modifications: Ensure the output has the correct number of channels
        # and use tanh activation, which is standard for GANs generating images.
        model.add(DepthwiseConv2D(kernel_size=3, padding="same"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same", activation="tanh"))
        
        # Print the summary of the model to check its architecture
        model.summary()

        # Inputs for the generator.
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        # Multiply the input noise with the label embedding to condition the generation process.
        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        # Return the model with noise and label as input and img as output.
        return Model([noise, label], img)

    def build_discriminator(self, input_shape):
        # sequential model
        model = Sequential()

        # totalpixels = input_shape[0] * input_shape[1] * input_shape[2]

        # # add reshape layer
        # model.add(Dense(totalpixels))
        # model.add(Reshape(input_shape=totalpixels, target_shape=input_shape))

        if self.use_HS_CNN:
            # what if the HS_CNN architecture is used as the discriminator?
            model.add(DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, input_shape=input_shape, depth_multiplier=3))  # depth_multiplier to expand channels
            model.add(Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True))
            model.add(ReLU())
            model.add(AveragePooling2D(pool_size=(4, 4), strides=(4, 4)))
            model.add(BatchNormalization())
                
            # Depthwise Convolution Block 2
            model.add(DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3))  # depth_multiplier to expand channels
            model.add(Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True))
            model.add(ReLU())
            model.add(AveragePooling2D(pool_size=(4, 4), strides=(4, 4)))
            model.add(BatchNormalization())
                
            # Depthwise Convolution Block 3
            model.add(DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3))  # depth_multiplier to expand channels
            model.add(Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True))
            model.add(ReLU())
            model.add(BatchNormalization())
            model.add(GlobalAveragePooling2D())
            model.add(Activation('sigmoid'))
                        
            # Fully Connected Layer
            model.add(BatchNormalization())
            model.add(Dense(50, use_bias=True))
            model.add(ReLU())
        else: 
            # # Convolutional layers for downsampling
            # # model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=input_shape, padding="same"))
            # model.add(DepthwiseConv2D(kernel_size=3, strides=2, input_shape=input_shape, padding="same"))
            # model.add(LeakyReLU(alpha=0.2))
            # model.add(Dropout(0.25))
            # # 
            # # model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
            # model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="same"))
            # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
            # model.add(BatchNormalization(momentum=0.8))
            # model.add(LeakyReLU(alpha=0.2))
            # model.add(Dropout(0.25))
            # # 
            # # model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
            # model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="same"))
            # model.add(LeakyReLU(alpha=0.2))
            # model.add(Dropout(0.25))
            # model.add(BatchNormalization(momentum=0.8))
            # # 
            # # model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
            # model.add(DepthwiseConv2D(kernel_size=3, strides=1, padding="same"))
            # model.add(LeakyReLU(alpha=0.2))
            # model.add(Dropout(0.25))
            pass

        # Flatten and final dense layer for classification
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=input_shape)
        label = Input(shape=(1,))
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(input_shape))(label))
        flat_img = Flatten()(img)

        embedded_input = multiply([flat_img, label_embedding])

        # reshape the input to the desired shape
        model_input = Reshape(input_shape)(embedded_input)

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size, sample_interval, save_interval, save_dir="./", x = None, y = None, datagen = None, use_multiprocessing = False, workers = 1):
        # if x and y are not provided skip initialising 
        if x is None and y is None:
            pass
        else:
            x_train = x
            y_train = y.reshape(-1,1)

        # 
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        for epoch in range(epochs):
            #####################
            # train discriminator
            #####################

            # use the datagen to generate a batch of images
            if datagen is not None:
                imgs, labels = datagen.__getitem__(0)
            else:
                # select a random batch of images
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                imgs = x_train[idx]
                labels = y_train[idx]

            # sample noise as generator input
            noise = np.random.normal(0,1,(batch_size, self.latent_dim))

            # generate a batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # train the discriminator

            # d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_fit_real = self.discriminator.fit([imgs, labels], valid, use_multiprocessing=use_multiprocessing, workers=workers)
            # the output of fit function is a history object, hence we need to extract the loss and accuracy from the history object
            d_loss_real = list(d_fit_real.history.values())

            # d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_fit_fake = self.discriminator.fit([gen_imgs, labels], fake, use_multiprocessing=use_multiprocessing, workers=workers)
            d_loss_fake = list(d_fit_fake.history.values())
            # 
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #####################
            # train the generator
            #####################

            # condition on labels
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1,1)

            # train the generator
            # g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
            g_fit = self.combined.fit([noise, sampled_labels], valid, use_multiprocessing=use_multiprocessing, workers=workers)
            g_loss = g_fit.history['loss'][-1]


            # print the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # call on_epoch_end method of the datagen
            if datagen is not None:
                datagen.on_epoch_end()

            # if at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            if epoch % save_interval == 0:
                self.save_model(save_dir)

    def sample_images(self, epoch):
        #  number of rows will be 5 
        r, c = 1, self.num_classes 
        # number of rows will be 5 
        figrows = 5
        skip = self.channels // figrows

        noise = np.random.normal(0,1,(r*c, self.latent_dim))
        sampled_labels = np.arange(0,self.num_classes).reshape(-1,1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(figrows,c)
        for i, ch in zip(range(figrows), range(0, self.channels, skip)):
            for j in range(c):
                # show a sample of channels from the generated images
                # out of all the channels in the image onlyshow a sample of 5 channels 
                axs[i,j].imshow(gen_imgs[j,:,:,ch])
                axs[i,j].set_title(f"class: {j}, channel: {ch}")
                axs[i,j].axis('off')
        fig.savefig(f"{self.working_dir}images/%d.png" % epoch)
        plt.close()

    def save_model(self, save_dir="gan_models/"):
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        gen_filename = f"{save_dir}generator_{current_date}.h5"
        self.generator.save(gen_filename)
        dis_filename = f"{save_dir}discriminator_{current_date}.h5"
        self.discriminator.save(dis_filename)
        comb_filename = f"{save_dir}combined_{current_date}.h5"
        self.combined.save(comb_filename)

    def load_model(self, gen_filename, dis_filename, comb_filename):
        self.generator = keras.models.load_model(gen_filename)
        self.discriminator = keras.models.load_model(dis_filename)
        self.combined = keras.models.load_model(comb_filename)