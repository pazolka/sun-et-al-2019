from keras.models import Model
from keras.layers import Conv2D, Input, Masking, Concatenate

def stage1(filters):
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
    model = Model(inputs=[input1, input2, input3], outputs=merged_features)
    return model
