from keras.models import Model
from keras.layers import Conv2D, Input, Masking, Concatenate, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Dense, Reshape
import keras.applications

def unet_full(filters):
	input1 = Input((128,128,3))
	conv1 = Conv2D(filters, 3, activation = 'relu', padding = 'same')(input1)
	input2 = Input((128,128,3))
	conv2 = Conv2D(filters,3, activation = 'relu', padding = 'same')(input2)
	input3 = Input((128,128,3))
	conv3 = Conv2D(filters, 3, activation = 'relu', padding = 'same')(input3)
	
	merged_features = Concatenate()([conv1, conv2, conv3])
	
	conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same')(merged_features)
	conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
	conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(pool1)
	conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	
	conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool2)
	conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	
	conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool3)
	conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	
	conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool4)
	conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv5)
	
	up6 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv5))
	merge6 = Concatenate(axis = 3)([conv4,up6])
	conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge6)
	conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv6)
	
	up7 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
	merge7 = Concatenate(axis = 3)([conv3,up7])
	conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge7)
	conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv7)
	
	up8 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
	merge8 = Concatenate(axis = 3)([conv2,up8])
	conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge8)
	conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv8)
	
	up9 = Conv2D(32, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
	merge9 = Concatenate(axis = 3)([conv1,up9])
	conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same')(merge9)
	conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv9)
	conv10 = Conv2D(1,1, 1, activation = 'linear')(conv9)
	
	model = Model(inputs = [input1, input2, input3], outputs = conv10)
	
	return model

def segnetlite_full(filters):
	input1 = Input((128,128,3))
	conv1 = Conv2D(filters, 3, activation = 'relu', padding = 'same')(input1)
	input2 = Input((128,128,3))
	conv2 = Conv2D(filters,3, activation = 'relu', padding = 'same')(input2)
	input3 = Input((128,128,3))
	conv3 = Conv2D(filters, 3, activation = 'relu', padding = 'same')(input3)
	
	merged_features = Concatenate()([conv1, conv2, conv3])

	conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same')(merged_features)
	conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same')(conv1)
	conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv2)
	conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv3)
	conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv4)
	conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv5)

	conv7 = Conv2DTranspose(128, 3, activation = 'leaky_relu', padding = 'same')(conv6)
	
	merge7 = Concatenate(axis = 3)([conv5,conv7])
	conv8 = Conv2DTranspose(64, 3, activation = 'leaky_relu', padding = 'same')(merge7)

	merge8 = Concatenate(axis = 3)([conv4,conv8])
	conv9 = Conv2DTranspose(64, 3, activation = 'leaky_relu', padding = 'same')(merge8)

	merge9 = Concatenate(axis = 3)([conv3,conv9])
	conv10 = Conv2DTranspose(32, 3, activation = 'leaky_relu', padding = 'same')(merge9)

	merge10 = Concatenate(axis = 3)([conv2,conv10])

	conv11 = Conv2DTranspose(16, 3, activation = 'leaky_relu', padding = 'same')(merge10)
	merge11 = Concatenate(axis = 3)([conv1,conv11])

	conv12 = Conv2D(1,1, 1, activation = 'linear')(merge11)

	model = Model(inputs = [input1, input2, input3], outputs = conv12)
	
	return model

def vgg16_full():
    input1 = Input((128,128,3))
    #input1 = keras.applications.vgg16.preprocess_input(input1)
    conv1 = Conv2D(1, 3, activation = 'relu', padding = 'same')(input1)
    
    input2 = Input((128,128,3))
    #input2 = keras.applications.vgg16.preprocess_input(input2)
    conv2 = Conv2D(1, 3, activation = 'relu', padding = 'same')(input2)

    input3 = Input((128,128,3))
    #input3 = keras.applications.vgg16.preprocess_input(input3)
    conv3 = Conv2D(1, 3, activation = 'relu', padding = 'same')(input3)
    
    merged_features = Concatenate()([conv1, conv2, conv3])

    input_shape = (128,128,3)

    base = keras.applications.VGG16(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape)
	
    for layer in base.layers:
        layer.trainable = False

    top_layer = base(merged_features)

    # conv1 = Conv2D(64, 3, padding="same", activation="relu")(merged_featured)
    # conv2 = Conv2D(64, 3, padding="same", activation="relu")(conv1)
    # pool1 = MaxPooling2D((2,2), (2,2))(conv2)

    # conv3 = Conv2D(128, 3, padding="same", activation="relu")(pool1)
    # conv4 = Conv2D(128, 3, padding="same", activation="relu")(conv3)
    # pool2 = MaxPooling2D((2,2),(2,2))(conv4)

    # conv5 = Conv2D(256, 3, padding="same", activation="relu")(pool2)
    # conv6 = Conv2D(256, 3, padding="same", activation="relu")(conv5)
    # conv7 = Conv2D(256, 3, padding="same", activation="relu")(conv6)
    # pool3 = MaxPooling2D((2,2), (2,2))(conv7)

    # conv8 = Conv2D(512, 3, padding="same", activation="relu")(pool3)
    # conv9 = Conv2D(512, 3, padding="same", activation="relu")(conv8)
    # conv10 = Conv2D(512, 3, padding="same", activation="relu")(conv9)
    # pool4 = MaxPooling2D((2,2), (2,2))(conv10)

    # conv11 = Conv2D(512, 3, padding="same", activation="relu")(pool4)
    # conv12 = Conv2D(512, 3, padding="same", activation="relu")(conv11)
    # conv13 = Conv2D(512, 3, padding="same", activation="relu")(conv12)
    # pool5 = MaxPooling2D((2,2),(2,2))(conv13)
	
    flatten = Flatten()(top_layer)
    dense = Dense(128*128, activation='linear')(flatten)
    output = Reshape((128, 128, 1))(dense)
    
    model = Model(inputs = [input1, input2, input3], outputs = output)

    return model