import numpy as np
import tensorflow as tf

# Define the custom metrics
def custom_nse(grace, gldas, y_pred):
    grace = grace[..., np.newaxis]
    gldas = gldas[..., np.newaxis]
    spatial_mean_per_ex_grace = tf.reduce_mean(grace, axis=(1,2))
    spatial_mean_per_ex_gldas = tf.reduce_mean(gldas, axis=(1,2))
    spatial_mean_per_ex_pred = tf.reduce_mean(y_pred, axis=(1,2))
    numerator = tf.reduce_sum(tf.square(spatial_mean_per_ex_grace - spatial_mean_per_ex_gldas + spatial_mean_per_ex_pred))
    denominator = tf.reduce_sum(tf.square(spatial_mean_per_ex_grace - tf.reduce_mean(spatial_mean_per_ex_grace, axis=0)))
    return 1 - numerator / (denominator + tf.keras.backend.epsilon())

def custom_corr(grace, gldas, y_pred):
    # x is GRACE TWSA
    x = grace[..., np.newaxis]
    # y is CNN-corrected TWSA
    y = gldas[..., np.newaxis]
    y = y - y_pred
    mx = tf.reduce_mean(x)
    my = tf.reduce_mean(y)
    x_m, y_m = x - mx, y - my
    r_num = tf.reduce_sum(x_m * y_m)
    x_square_sum = tf.reduce_sum(x_m * x_m)
    y_square_sum = tf.reduce_sum(y_m * y_m)
    r_den = tf.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + tf.keras.backend.epsilon())
    return r
