import numpy as np
import tensorflow as tf

# Define the custom loss function
def custom_loss(y_true, y_pred):
    # Mean Squared Error (MSE)
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)

    # Nash-Sutcliffe Efficiency (NSE) component
    numerator = tf.reduce_sum(tf.abs(y_pred - y_true))
    denominator = tf.reduce_sum(tf.abs(y_true - tf.reduce_mean(y_true, axis=0)))
    nse = numerator / (denominator + tf.keras.backend.epsilon())

    # Combine MSE and NSE
    combined_loss = 0.5 * mse + 0.5 * nse
    return combined_loss