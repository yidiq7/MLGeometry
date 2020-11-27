import tensorflow as tf
import tensorflow.python.keras.backend as K

__all__ = ['weighted_MAPE','weighted_MSE','max_error','MAPE_plus_max_error']

def weighted_MAPE(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum(K.abs(y_true - y_pred) / y_true * weights)

def weighted_MSE(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum(tf.square(y_pred / y_true - 1) * weights)

def max_error(y_true, y_pred, mass):
    return tf.math.reduce_max(K.abs(y_true - y_pred) / y_true)

def MAPE_plus_max_error(y_true, y_pred, mass):
    return 1*max_error(y_true, y_pred, mass) + weighted_MAPE(y_true, y_pred, mass)

