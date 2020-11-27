from tensorflow import keras
from tensorflow.python.keras import activations
import numpy as np 
import tensorflow as tf

__all__ = ['Bihomogeneous','Bihomogeneous_k2','Bihomogeneous_k3',
           'Bihomogeneous_k4','Dense','WidthOneDense']

class Bihomogeneous(keras.layers.Layer):
    '''A layer transform zi to zi*zjbar'''
    def __init__(self):
        super(Bihomogeneous, self).__init__()
        
    def call(self, inputs):
        zzbar = tf.einsum('ai,aj->aij', inputs, tf.math.conj(inputs))
        zzbar = tf.linalg.band_part(zzbar, 0, -1)
        zzbar = tf.reshape(zzbar, [-1, 25])
        zzbar = tf.concat([tf.math.real(zzbar), tf.math.imag(zzbar)], axis=1)
        return remove_zero_entries(zzbar)
        
class Bihomogeneous_k2(keras.layers.Layer):
    '''A layer transform zi to symmetrized zi1*zi2, then to zi1*zi2 * zi1zi2bar'''
    def __init__(self):
        super(Bihomogeneous_k2, self).__init__()
        
    def call(self, inputs):
        # zi to zi1*zi2 
        zz = tf.einsum('ai,aj->aij', inputs, inputs)
        zz = tf.linalg.band_part(zz, 0, -1) # zero below upper triangular
        zz = tf.reshape(zz, [-1, 5**2])
        zz = tf.reshape(remove_zero_entries(zz), [-1, 15])
     
        # zi1*zi2 to zzbar
        zzbar = tf.einsum('ai,aj->aij', zz, tf.math.conj(zz))
        zzbar = tf.linalg.band_part(zzbar, 0, -1)
        zzbar = tf.reshape(zzbar, [-1, 15**2])
        zzbar = tf.concat([tf.math.real(zzbar), tf.math.imag(zzbar)], axis=1)
        return remove_zero_entries(zzbar)


class Bihomogeneous_k3(keras.layers.Layer):
    '''A layer transform zi to symmetrized zi1*zi2*zi3, then to zzbar'''
    def __init__(self):
        super(Bihomogeneous_k3, self).__init__()
        
    def call(self, inputs):
        zz = tf.einsum('ai,aj,ak->aijk', inputs, inputs, inputs)
        zz = tf.linalg.band_part(zz, 0, -1) # keep upper triangular 2/3
        zz = tf.transpose(zz, perm=[0, 3, 1, 2])
        zz = tf.linalg.band_part(zz, 0, -1) # keep upper triangular 1/2
        zz = tf.transpose(zz, perm=[0, 2, 3, 1])
        zz = tf.reshape(zz, [-1, 5**3]) 
        zz = tf.reshape(remove_zero_entries(zz), [-1, 35])

        zzbar = tf.einsum('ai,aj->aij', zz, tf.math.conj(zz))
        zzbar = tf.linalg.band_part(zzbar, 0, -1)
        zzbar = tf.reshape(zzbar, [-1, 35**2])
        zzbar = tf.concat([tf.math.real(zzbar), tf.math.imag(zzbar)], axis=1)
        return remove_zero_entries(zzbar)

class Bihomogeneous_k4(keras.layers.Layer):
    '''A layer transform zi to symmetrized zi1*zi2*zi3*zi4, then to zzbar'''
    def __init__(self):
        super(Bihomogeneous_k4, self).__init__()
        
    def call(self, inputs):
        zz = tf.einsum('ai,aj,ak,al->aijkl', inputs, inputs, inputs, inputs)
        zz = tf.linalg.band_part(zz, 0, -1) 
        zz = tf.transpose(zz, perm=[0, 4, 1, 2, 3])
        zz = tf.linalg.band_part(zz, 0, -1) 
        zz = tf.transpose(zz, perm=[0, 4, 1, 2, 3]) # 3412
        zz = tf.linalg.band_part(zz, 0, -1) 
        zz = tf.reshape(zz, [-1, 5**4]) 
        zz = tf.reshape(remove_zero_entries(zz), [-1, 70])

        zzbar = tf.einsum('ai,aj->aij', zz, tf.math.conj(zz))
        zzbar = tf.linalg.band_part(zzbar, 0, -1)
        zzbar = tf.reshape(zzbar, [-1, 70**2])
        zzbar = tf.concat([tf.math.real(zzbar), tf.math.imag(zzbar)], axis=1)
        return remove_zero_entries(zzbar)

def remove_zero_entries(x):
    x = tf.transpose(x)
    intermediate_tensor = tf.reduce_sum(tf.abs(x), 1)
    bool_mask = tf.squeeze(tf.math.logical_not(tf.math.less(intermediate_tensor, 1e-3)))
    x = tf.boolean_mask(x, bool_mask)
    x = tf.transpose(x)
    return x

class Dense(keras.layers.Layer):
    def __init__(self, input_dim, units, activation=None, trainable=True):
        super(Dense, self).__init__()
        w_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)
        self.w = tf.Variable(
            #initial_value=tf.math.abs(w_init(shape=(input_dim, units), dtype='float32')),
            initial_value=w_init(shape=(input_dim, units), dtype='float32'),
            trainable=trainable,
        )
        self.activation =  activations.get(activation)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w))

class WidthOneDense(keras.layers.Layer):
    '''
    Usage: layer = WidthOneDense(n**2, 1)
           where n is the number of sections for different ks
           n = 5 for k = 1
           n = 15 for k = 2
           n = 35 for k = 3
    This layer is used directly after Bihomogeneous_k layers to sum over all 
    the terms in the previous layer. The weights are initialized so that the h
    matrix is a real identity matrix. The training does not work if they are randomly
    initialized.
    '''
    def __init__(self, input_dim, units, activation=None, trainable=True):
        super(WidthOneDense, self).__init__()
        dim = int(np.sqrt(input_dim))
        mask = tf.cast(tf.linalg.band_part(tf.ones([dim, dim]),0,-1), dtype=tf.bool)
        upper_tri = tf.boolean_mask(tf.eye(dim), mask)
        w_init = tf.reshape(tf.concat([upper_tri, tf.zeros(input_dim - len(upper_tri))], axis=0), [-1, 1])
        self.w = tf.Variable(
            initial_value=w_init,
            trainable=trainable,
        )
        self.activation =  activations.get(activation)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w))

