import tensorflow as tf
keras = tf.keras
from keras.models import Sequential
from keras.layers import DepthwiseConv2D, AveragePooling2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Conv2D, SeparableConvolution2D, Activation, Input, Add, Lambda, concatenate, Attention
from keras.activations import sigmoid
from keras.backend import resize_images
from keras_adabound import AdaBound
from keras.optimizers import Adam, RMSprop

def build_HS_CNN(input_shape):
    # Depthwise Convolution Block 1
    HS_CNN = Sequential()
    HS_CNN.add(DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, input_shape=input_shape, depth_multiplier=3))  # depth_multiplier to expand channels
    HS_CNN.add(Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True))
    HS_CNN.add(ReLU())
    HS_CNN.add(AveragePooling2D(pool_size=(4, 4), strides=(4, 4)))
    HS_CNN.add(BatchNormalization())
        
    # Depthwise Convolution Block 2
    HS_CNN.add(DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3))  # depth_multiplier to expand channels
    HS_CNN.add(Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True))
    HS_CNN.add(ReLU())
    HS_CNN.add(AveragePooling2D(pool_size=(4, 4), strides=(4, 4)))
    HS_CNN.add(BatchNormalization())
        
    # Depthwise Convolution Block 3
    HS_CNN.add(DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3))  # depth_multiplier to expand channels
    HS_CNN.add(Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True))
    HS_CNN.add(ReLU())
    HS_CNN.add(BatchNormalization())
    HS_CNN.add(GlobalAveragePooling2D())
        
    HS_CNN.add(Activation('sigmoid'))
        
    # Fully Connected Layer
    HS_CNN.add(BatchNormalization())
    HS_CNN.add(Dense(50, use_bias=True))
    HS_CNN.add(ReLU())
    HS_CNN.add(Dense(3, use_bias=True))
    HS_CNN.add(Activation('sigmoid'))

    # set model name 
    HS_CNN.__name__ = "HS_CNN"
    # HS_CNN.compile(optimizer=AdaBound(learning_rate=1e-2), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return HS_CNN

def build_MSDCNN(input_shape):
    input_layer = Input(shape=input_shape)

    # original scale pathway
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(input_layer)
    a = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a) 
    # Depthwise Convolution Block 2
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(a)  # depth_multiplier to expand channels
    a = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a)
    # Depthwise Convolution Block 3
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(a) # depth_multiplier to expand channels
    a = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = BatchNormalization()(a)
    a = GlobalAveragePooling2D()(a)

    # double scale pathway 
    def resize_double(input_img):
        return resize_images(input_img, height_factor=2, width_factor=2, data_format='channels_last')
    b = Lambda(resize_double)(input_layer)
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)
    b = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    # Depthwise Convolution Block 2
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)  # depth_multiplier to expand channels
    b = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    # Depthwise Convolution Block 3
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)  # depth_multiplier to expand channels
    b = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = BatchNormalization()(b)
    b = GlobalAveragePooling2D()(b)

    # half scale pathway 
    def resize_half(input_img):
        # return resize_images(input_img, height_factor=0.5, width_factor=0.5, data_format='channels_last')
        # get image shape 
        shape = input_img.shape
        x = shape[1]//2 
        y = shape[2]//2 
        return tf.image.resize(input_img, (x,y))
    c = Lambda(resize_half)(input_layer)
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)
    c = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    # Depthwise Convolution Block 2
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)  # depth_multiplier to expand channels
    c = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    # Depthwise Convolution Block 3
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)  # depth_multiplier to expand channels
    c = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = BatchNormalization()(c)
    c = GlobalAveragePooling2D()(c)

    # concatenate the three pathways
    x = concatenate([a, b, c])
    x = Activation('sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dense(150, use_bias=True)(x)
    x = ReLU()(x)
    x = Dense(3, use_bias=True)(x)
    x = Activation('sigmoid')(x)
    
    model = keras.Model(inputs=input_layer, outputs=x)
    model.__name__ = "MSDCNN"
    # model.compile(optimizer=Adam(learning_rate=1e-2, beta_1=0.5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_MSDCNNV(input_shape):
    input_layer = Input(shape=input_shape)

    # original scale pathway
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(input_layer)
    a = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a) 
    # Depthwise Convolution Block 2
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(a)  # depth_multiplier to expand channels
    a = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a)
    # Depthwise Convolution Block 3
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(a) # depth_multiplier to expand channels
    a = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = BatchNormalization()(a)
    a = GlobalAveragePooling2D()(a)
    a = Activation('sigmoid')(a)
    a = BatchNormalization()(a)
    a = Dense(50, use_bias=True)(a)
    a = ReLU()(a)
    a = Dense(3, use_bias=True)(a)
    a = Activation('sigmoid')(a)

    # double scale pathway 
    def resize_double(input_img):
        return resize_images(input_img, height_factor=2, width_factor=2, data_format='channels_last')
    b = Lambda(resize_double)(input_layer)
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)
    b = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    # Depthwise Convolution Block 2
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)  # depth_multiplier to expand channels
    b = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    # Depthwise Convolution Block 3
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)  # depth_multiplier to expand channels
    b = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = BatchNormalization()(b)
    b = GlobalAveragePooling2D()(b)
    b = Activation('sigmoid')(b)
    b = BatchNormalization()(b)
    b = Dense(50, use_bias=True)(b)
    b = ReLU()(b)
    b = Dense(3, use_bias=True)(b)
    b = Activation('sigmoid')(b)

    # half scale pathway 
    def resize_half(input_img):
        # return resize_images(input_img, height_factor=0.5, width_factor=0.5, data_format='channels_last')
        # get image shape 
        shape = input_img.shape
        x = shape[1]//2 
        y = shape[2]//2 
        return tf.image.resize(input_img, (x,y))
    c = Lambda(resize_half)(input_layer)
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)
    c = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    # Depthwise Convolution Block 2
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)  # depth_multiplier to expand channels
    c = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    # Depthwise Convolution Block 3
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)  # depth_multiplier to expand channels
    c = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = BatchNormalization()(c)
    c = GlobalAveragePooling2D()(c)
    c = Activation('sigmoid')(c)
    c = BatchNormalization()(c)
    c = Dense(50, use_bias=True)(c)
    c = ReLU()(c)
    c = Dense(3, use_bias=True)(c)
    c = Activation('sigmoid')(c)

    # concatenate the three pathways
    x = Add()([a,b,c])
    # voting mechanism
    x = Activation('softmax')(x)
    
    model = keras.Model(inputs=input_layer, outputs=x)
    model.__name__ = "MSDCNNV"
    # model.compile(optimizer=Adam(learning_rate=1e-2, beta_1=0.5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_MKDCNN(input_shape):
    # Input Layer
    input_layer = Input(shape=input_shape)

    # Fine Scale Path
    a = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    a = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a)
    a = ReLU()(a)
    
    # Medium Scale Path
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    b = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    b = ReLU()(b)

    # Coarse Scale Path
    c = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    c = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    c = ReLU()(c)

    # Feature Fusion
    x = concatenate([a, b, c])  
    x = GlobalAveragePooling2D()(x)
    x = Activation('sigmoid')(x)  
    x = BatchNormalization()(x)
    x = Dense(50, use_bias=True)(x)
    x = ReLU()(x)

    # Output Layer
    x = Dense(3, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)
    model.__name__ = "MKDCNN"
    # model.compile(optimizer=Adam(learning_rate=1e-2, beta_1=0.5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_MKDCNN_Attention(input_shape):
    # Input Layer
    input_layer = Input(shape=input_shape)

    # Fine Scale Path
    a = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    a = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a)
    a = ReLU()(a)
    
    # Medium Scale Path
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    b = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    b = ReLU()(b)

    # Coarse Scale Path
    c = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    c = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    c = ReLU()(c)

    # Feature Fusion
    x = Attention()([a, b, c])  
    x = GlobalAveragePooling2D()(x)
    x = Activation('sigmoid')(x)  
    x = BatchNormalization()(x)
    x = Dense(50, use_bias=True)(x)
    x = ReLU()(x)

    # Output Layer
    x = Dense(3, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)
    model.__name__ = "MKDCNN_Attention"
    # model.compile(optimizer=Adam(learning_rate=1e-2, beta_1=0.5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_MSDCNN_Attention(input_shape):
    # multi scale depth wise convolutional neural network
    input_layer = Input(shape=input_shape)

    # original scale pathway
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(input_layer)
    a = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a) 
    # Depthwise Convolution Block 2
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(a)  # depth_multiplier to expand channels
    a = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a)
    # Depthwise Convolution Block 3
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(a) # depth_multiplier to expand channels
    a = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = BatchNormalization()(a)
    a = GlobalAveragePooling2D()(a)

    # double scale pathway 
    def resize_double(input_img):
        return resize_images(input_img, height_factor=2, width_factor=2, data_format='channels_last')
    b = Lambda(resize_double)(input_layer)
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)
    b = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    # Depthwise Convolution Block 2
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)  # depth_multiplier to expand channels
    b = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    # Depthwise Convolution Block 3
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)  # depth_multiplier to expand channels
    b = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = BatchNormalization()(b)
    b = GlobalAveragePooling2D()(b)

    # half scale pathway 
    def resize_half(input_img):
        # return resize_images(input_img, height_factor=0.5, width_factor=0.5, data_format='channels_last')
        # get image shape 
        shape = input_img.shape
        x = shape[1]//2 
        y = shape[2]//2 
        return tf.image.resize(input_img, (x,y))
    c = Lambda(resize_half)(input_layer)
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)
    c = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    # Depthwise Convolution Block 2
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)  # depth_multiplier to expand channels
    c = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    # Depthwise Convolution Block 3
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)  # depth_multiplier to expand channels
    c = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = BatchNormalization()(c)
    c = GlobalAveragePooling2D()(c)

    # concatenate the three pathways
    x = Attention()([a, b, c])
    x = Activation('sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dense(50, use_bias=True)(x)
    x = ReLU()(x)
    x = Dense(3, use_bias=True)(x)
    x = Activation('sigmoid')(x)
    
    model = keras.Model(inputs=input_layer, outputs=x)
    model.__name__ = "MSDCNN_Attention"
    return model

def build_MKDCNN_params(params, input_shape):
    # Input Layer
    input_layer = Input(shape=input_shape)

    # unpack params
    k1, c1, k2, c2, k3, c3, lr = params

    # convert to integers 
    k1 = int(k1)
    c1 = int(c1)
    k2 = int(k2)
    c2 = int(c2)
    k3 = int(k3)
    c3 = int(c3)

    # Fine Scale Path
    a = DepthwiseConv2D(kernel_size=(k1, k1), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    a = Conv2D(c1, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a)
    a = ReLU()(a)
    
    # Medium Scale Path
    b = DepthwiseConv2D(kernel_size=(k2, k2), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    b = Conv2D(c2, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    b = ReLU()(b)

    # Coarse Scale Path
    c = DepthwiseConv2D(kernel_size=(k3, k3), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    c = Conv2D(c3, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    c = ReLU()(c)

    # Feature Fusion
    x = concatenate([a, b, c])  
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)  
    x = BatchNormalization()(x)
    x = Dense(50, use_bias=True)(x)
    x = ReLU()(x)

    # Output Layer
    x = Dense(3, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)
    model.__name__ = "MKDCNN_OPTIMIZED"
    model.compile(optimizer=RMSprop(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_MKDCNN_Attention_params(params, input_shape):
    # Input Layer
    input_layer = Input(shape=input_shape)

    # unpack params
    k1, c1, k2, c2, k3, c3, lr = params

    # convert to integers 
    k1 = int(k1)
    c1 = int(c1)
    k2 = int(k2)
    c2 = int(c2)
    k3 = int(k3)
    c3 = int(c3)

    # Fine Scale Path
    a = DepthwiseConv2D(kernel_size=(k1, k1), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    a = Conv2D(c1, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a)
    a = ReLU()(a)
    
    # Medium Scale Path
    b = DepthwiseConv2D(kernel_size=(k2, k2), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    b = Conv2D(c2, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    b = ReLU()(b)

    # Coarse Scale Path
    c = DepthwiseConv2D(kernel_size=(k3, k3), strides=(1, 1), padding='same', use_bias=False)(input_layer)
    c = Conv2D(c3, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    c = ReLU()(c)

    # Feature Fusion
    x = Attention()([a, b, c])  
    x = GlobalAveragePooling2D()(x)
    x = Activation('sigmoid')(x)  
    x = BatchNormalization()(x)
    x = Dense(50, use_bias=True)(x)
    x = ReLU()(x)

    # Output Layer
    x = Dense(3, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)
    model.__name__ = "MKDCNN_Attention_OPTIMIZED"
    model.compile(optimizer=RMSprop(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model 

def build_MSDCNN_params(params, input_shape):
    input_layer = Input(shape=input_shape)

    # original scale pathway
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(input_layer)
    a = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a) 
    # Depthwise Convolution Block 2
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(a)  # depth_multiplier to expand channels
    a = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a)
    # Depthwise Convolution Block 3
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(a) # depth_multiplier to expand channels
    a = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = BatchNormalization()(a)
    a = GlobalAveragePooling2D()(a)

    # double scale pathway 
    def resize_double(input_img):
        return resize_images(input_img, height_factor=2, width_factor=2, data_format='channels_last')
    b = Lambda(resize_double)(input_layer)
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)
    b = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    # Depthwise Convolution Block 2
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)  # depth_multiplier to expand channels
    b = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    # Depthwise Convolution Block 3
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)  # depth_multiplier to expand channels
    b = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = BatchNormalization()(b)
    b = GlobalAveragePooling2D()(b)

    # half scale pathway 
    def resize_half(input_img):
        # return resize_images(input_img, height_factor=0.5, width_factor=0.5, data_format='channels_last')
        # get image shape 
        shape = input_img.shape
        x = shape[1]//2 
        y = shape[2]//2 
        return tf.image.resize(input_img, (x,y))
    c = Lambda(resize_half)(input_layer)
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)
    c = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    # Depthwise Convolution Block 2
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)  # depth_multiplier to expand channels
    c = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    # Depthwise Convolution Block 3
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)  # depth_multiplier to expand channels
    c = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = BatchNormalization()(c)
    c = GlobalAveragePooling2D()(c)

    # concatenate the three pathways
    x = concatenate([a, b, c])
    x = Activation('sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dense(150, use_bias=True)(x)
    x = ReLU()(x)
    x = Dense(3, use_bias=True)(x)
    x = Activation('sigmoid')(x)
    
    model = keras.Model(inputs=input_layer, outputs=x)
    model.__name__ = "MSDCNN_OPTIMIZED"
    model.compile(optimizer=Adam(learning_rate=1e-2, beta_1=0.5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_MSDCNN_Attention_params(params, input_shape):
    # multi scale depth wise convolutional neural network
    input_layer = Input(shape=input_shape)

    # original scale pathway
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(input_layer)
    a = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a) 
    # Depthwise Convolution Block 2
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(a)  # depth_multiplier to expand channels
    a = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(a)
    a = BatchNormalization()(a)
    # Depthwise Convolution Block 3
    a = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(a) # depth_multiplier to expand channels
    a = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(a)
    a = ReLU()(a)
    a = BatchNormalization()(a)
    a = GlobalAveragePooling2D()(a)

    # double scale pathway 
    def resize_double(input_img):
        return resize_images(input_img, height_factor=2, width_factor=2, data_format='channels_last')
    b = Lambda(resize_double)(input_layer)
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)
    b = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    # Depthwise Convolution Block 2
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)  # depth_multiplier to expand channels
    b = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(b)
    b = BatchNormalization()(b)
    # Depthwise Convolution Block 3
    b = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(b)  # depth_multiplier to expand channels
    b = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(b)
    b = ReLU()(b)
    b = BatchNormalization()(b)
    b = GlobalAveragePooling2D()(b)

    # half scale pathway 
    def resize_half(input_img):
        # return resize_images(input_img, height_factor=0.5, width_factor=0.5, data_format='channels_last')
        # get image shape 
        shape = input_img.shape
        x = shape[1]//2 
        y = shape[2]//2 
        return tf.image.resize(input_img, (x,y))
    c = Lambda(resize_half)(input_layer)
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)
    c = Conv2D(25, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    # Depthwise Convolution Block 2
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)  # depth_multiplier to expand channels
    c = Conv2D(30, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(c)
    c = BatchNormalization()(c)
    # Depthwise Convolution Block 3
    c = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, depth_multiplier=3)(c)  # depth_multiplier to expand channels
    c = Conv2D(50, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(c)
    c = ReLU()(c)
    c = BatchNormalization()(c)
    c = GlobalAveragePooling2D()(c)

    # concatenate the three pathways
    x = Attention()([a, b, c])
    # x = GlobalAveragePooling2D()(x)
    x = Activation('sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dense(50, use_bias=True)(x)
    x = ReLU()(x)
    x = Dense(3, use_bias=True)(x)
    x = Activation('sigmoid')(x)
    
    model = keras.Model(inputs=input_layer, outputs=x)
    model.__name__ = "MSDCNN_Attention_OPTIMIZED"
    model.compile(optimizer=Adam(learning_rate=1e-2, beta_1=0.5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# EOF