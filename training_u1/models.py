import tensorflow as tf
from MLGeometry.u1equivariant import U1EquivariantLayer

__all__ = ['u1_model']

kINPUT_DIM = 10
kPAIR_DIM = 50

class u1_model_relu(tf.keras.Model):

    def __init__(self, n_units, m_units):
        super().__init__()
        assert len(n_units) > 0
        assert len(m_units) > 0

        inner_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(
                n_units[i],
                input_shape=((kINPUT_DIM, ) if i == 0 else None),
                activation=('relu' if i < len(n_units) - 1 else None)
            )
            for i in range(len(n_units))
        ])

        self.u1_layer = U1EquivariantLayer(inner_layers)

        self.outer_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(
                m_units[i],
                activation=('relu' if i < len(m_units) - 1 else None)
            )
            for i in range(len(m_units))
        ])

    def call(self, inputs):
        assert len(inputs.shape) == 3
        ps, bs = inputs.shape[:2]
        assert ps == kPAIR_DIM
        x = tf.reshape(inputs, [ps * bs, -1])
        x = self.u1_layer(x)
        x = self.outer_layers(x)
        x = tf.reshape(x, [ps, bs, -1])
        x = tf.math.log(x)
        return x
