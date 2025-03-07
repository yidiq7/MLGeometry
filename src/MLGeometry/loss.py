import tensorflow as tf
import keras

__all__ = ['weighted_MAPE','weighted_MSE','max_error','MAPE_plus_max_error']

@keras.saving.register_keras_serializable(package="MLGeometry")
def weighted_MAPE(y_true, y_pred, mass):
    weights = mass / tf.reduce_sum(mass)
    return tf.reduce_sum(tf.abs(y_true - y_pred) / y_true * weights)


@keras.saving.register_keras_serializable(package="MLGeometry")
def weighted_MSE(y_true, y_pred, mass):
    weights = mass / tf.reduce_sum(mass)
    return tf.reduce_sum(tf.square(y_pred / y_true - 1) * weights)


@keras.saving.register_keras_serializable(package="MLGeometry")
def max_error(y_true, y_pred, mass):
    return tf.math.reduce_max(tf.abs(y_true - y_pred) / y_true)


@keras.saving.register_keras_serializable(package="MLGeometry")
def MAPE_plus_max_error(y_true, y_pred, mass):
    return 1*max_error(y_true, y_pred, mass) + weighted_MAPE(y_true, y_pred, mass)

