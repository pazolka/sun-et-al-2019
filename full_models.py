from keras.models import Model
from keras.layers import Conv2D, Input, Masking, Concatenate, MaxPooling2D, UpSampling2D, Conv2DTranspose

def unet_full(filters):
	input1 = Input((128,128,3))
	masked_input1 = Masking(mask_value=1e-7)(input1)
	conv1 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(masked_input1)
	input2 = Input((128,128,3))
	masked_input2 = Masking(mask_value=1e-7)(input2)
	conv2 = Conv2D(filters,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(masked_input2)
	input3 = Input((128,128,3))
	masked_input3 = Masking(mask_value=1e-7)(input3)
	conv3 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(masked_input3)
	
	merged_features = Concatenate()([conv1, conv2, conv3])

	conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merged_features)
	conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	
	conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	
	conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	
	conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
	conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
	
	up6 = UpSampling2D(size = (2,2))(conv5)
	merge6 = Concatenate(axis = 3)([conv4,up6])
	conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
	
	up7 = UpSampling2D(size = (2,2))(conv6)
	merge7 = Concatenate(axis = 3)([conv3,up7])
	conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
	
	up8 = UpSampling2D(size = (2,2))(conv7)
	merge8 = Concatenate(axis = 3)([conv2,up8])
	conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
	
	up9 = UpSampling2D(size = (2,2))(conv8)
	merge9 = Concatenate(axis = 3)([conv1,up9])
	conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv10 = Conv2D(1,1, 1, activation = 'linear')(conv9)
	
	model = Model(inputs = [input1, input2, input3], outputs = conv10)
	
	return model

def segnetlite_full(filters):
	input1 = Input((128,128,3))
	masked_input1 = Masking(mask_value=1e-7)(input1)
	conv1 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(masked_input1)
	input2 = Input((128,128,3))
	masked_input2 = Masking(mask_value=1e-7)(input2)
	conv2 = Conv2D(filters,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(masked_input2)
	input3 = Input((128,128,3))
	masked_input3 = Masking(mask_value=1e-7)(input3)
	conv3 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(masked_input3)
	
	merged_features = Concatenate()([conv1, conv2, conv3])

	conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merged_features)
	conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

	merge7 = Concatenate(axis = 3)([conv5,conv6])
    conv7 = Conv2DTranspose(128, 3, activation = 'leaky_relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)

	merge8 = Concatenate(axis = 3)([conv4,conv7])
	conv8 = Conv2DTranspose(64, 3, activation = 'leaky_relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)

	merge9 = Concatenate(axis = 3)([conv3,conv8])
    conv9 = Conv2DTranspose(64, 3, activation = 'leaky_relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)

	merge9 = Concatenate(axis = 3)([conv2,conv9])
    conv10 = Conv2DTranspose(32, 3, activation = 'leaky_relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

	merge10 = Concatenate(axis = 3)([conv1,conv10])
    conv11 = Conv2DTranspose(16, 3, activation = 'leaky_relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
	conv12 = Conv2D(1,1, 1, activation = 'linear')(conv11)

	model = Model(inputs = [input1, input2, input3], outputs = conv12)
	
	return model
