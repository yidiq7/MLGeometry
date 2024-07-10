import tensorflow as tf
import numpy as np


class U1EquivariantLayer(tf.keras.Model):

    def __init__(self, layer, g_steps = 1, g_bias = False):
        super(U1EquivariantLayer, self).__init__()
        self.n_steps = g_steps
        self.gen_bias = g_bias
        self.layer = layer

        phi = tf.constant(np.linspace(
            start = 0,
            stop = 2*np.pi*(self.n_steps-1) / self.n_steps,
            num = self.n_steps,
        ), dtype=tf.float32)
        
        # self.phi = phi

        if self.gen_bias:
            # random number (0, 2*np.pi)
            bias = tf.random.uniform(shape=[1], minval=0, maxval=2 * np.pi / self.n_steps, dtype=tf.float32)
            phi = phi + bias

        self.group_elements = tf.constant(
            tf.math.exp(1j * tf.cast(phi, tf.complex64)),
            dtype=tf.complex64
        )

        assert self.group_elements.dtype == tf.complex64
        assert self.group_elements.shape[0] == self.n_steps

    def call(self, inputs):
        if len(inputs.shape) != 2:
            raise ValueError("Wrong shape expected (B, C) but it is ", inputs.shape)

        batch_size, dim = inputs.shape
        z = inputs
        z = z / tf.cast(tf.math.abs(z), tf.complex64)
        z = tf.multiply(self.group_elements[None, :, None],  z[:, None, :])
        # z = tf.reshape(z, [batch_size*self.n_steps, dim])
        z = tf.reshape(z, [-1, dim])
        z_real, z_imag = tf.math.real(z), tf.math.imag(z)
        z = tf.concat([z_real, z_imag], axis=1)

        z = self.layer(z)

        # z = tf.reshape(z, [batch_size, self.n_steps, -1])
        z = tf.reshape(z, [-1, self.n_steps, self.layer.output_shape[-1]])

        z = tf.math.reduce_mean(z, axis=1)
        z = tf.cast(z, tf.float32)
        return z