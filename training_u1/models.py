import tensorflow as tf
from MLGeometry import bihomoNN as bnn
from MLGeometry.u1equivariant import U1EquivariantLayer

__all__ = ['u1_model']




class u1_model_relu(tf.keras.Model):

    def __init__(self, n_units, m_units):
        super().__init__()
        assert len(n_units) > 0
        assert len(m_units) > 0

        input_dim = 10

        inner_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(
                n_units[i],
                input_shape=(None if i > 0 else (input_dim, )),
                activation=('relu' if i < len(n_units) - 1 else None)
            )
            for i in range(len(n_units))
        ])

        self.outer_u1_layers = tf.keras.Sequential([
            U1EquivariantLayer(inner_layers)
        ] + [
            tf.keras.layers.Dense(
                m_units[i],
                activation=('relu' if i < len(m_units) - 1 else None)
            )
            for i in range(len(m_units))
        ])

    def call(self, inputs):
        x = self.outer_u1_layers(inputs)
        x = tf.math.log(x)
        return x
