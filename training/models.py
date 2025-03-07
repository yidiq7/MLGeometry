import tensorflow as tf
from MLGeometry import bihomoNN as bnn

__all__ = ['zerolayer', 'onelayer', 'twolayers', 'threelayers', 'fourlayers', 
           'fivelayers','OuterProductNN_k2','OuterProductNN_k3','OuterProductNN_k4',
           'k2_twolayers', 'k2_threelayers','k4_onelayer','k4_twolayers']
 
class zerolayer(keras.Model):

    def __init__(self, n_units):
        super(zerolayer, self).__init__()
        self.bihomogeneous = bnn.Bihomogeneous()
        self.layer1 = bnn.WidthOneDense(25, 1)

    def call(self, inputs):
        x = self.bihomogeneous(inputs)
        x = self.layer1(x)
        x = tf.math.log(x)
        return x

class onelayer(keras.Model):

    def __init__(self, n_units):
        super(onelayer, self).__init__()
        self.bihomogeneous = bnn.Bihomogeneous()
        self.layer1 = bnn.SquareDense(25, n_units[0], activation=tf.square)
        self.layer2 = bnn.SquareDense(n_units[0], 1)

    def call(self, inputs):
        x = self.bihomogeneous(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = tf.math.log(x)
        return x


class twolayers(keras.Model):

    def __init__(self, n_units):
        super(twolayers, self).__init__()
        self.bihomogeneous = bnn.Bihomogeneous()
        self.layer1 = bnn.SquareDense(25, n_units[0], activation=tf.square)
        self.layer2 = bnn.SquareDense(n_units[0], n_units[1], activation=tf.square)
        self.layer3 = bnn.SquareDense(n_units[1], 1)

    def call(self, inputs):
        x = self.bihomogeneous(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = tf.math.log(x)
        return x


class threelayers(keras.Model):

    def __init__(self, n_units):
        super(threelayers, self).__init__()
        self.bihomogeneous = bnn.Bihomogeneous()
        self.layer1 = bnn.SquareDense(25, n_units[0], activation=tf.square)
        self.layer2 = bnn.SquareDense(n_units[0], n_units[1], activation=tf.square)
        self.layer3 = bnn.SquareDense(n_units[1], n_units[2], activation=tf.square)
        self.layer4 = bnn.SquareDense(n_units[2], 1)

    def call(self, inputs):
        x = self.bihomogeneous(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = tf.math.log(x)
        return x


class fourlayers(keras.Model):

    def __init__(self, n_units):
        super(fourlayers, self).__init__()
        self.bihomogeneous = bnn.Bihomogeneous()
        self.layer1 = bnn.SquareDense(25, n_units[0], activation=tf.square)
        self.layer2 = bnn.SquareDense(n_units[0], n_units[1], activation=tf.square)
        self.layer3 = bnn.SquareDense(n_units[1], n_units[2], activation=tf.square)
        self.layer4 = bnn.SquareDense(n_units[2], n_units[3], activation=tf.square)
        self.layer5 = bnn.SquareDense(n_units[3], 1)

    def call(self, inputs):
        x = self.bihomogeneous(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = tf.math.log(x)
        return x


class fivelayers(keras.Model):

    def __init__(self, n_units):
        super(fivelayers, self).__init__()
        self.bihomogeneous = bnn.Bihomogeneous()
        self.layer1 = bnn.SquareDense(25, n_units[0], activation=tf.square)
        self.layer2 = bnn.SquareDense(n_units[0], n_units[1], activation=tf.square)
        self.layer3 = bnn.SquareDense(n_units[1], n_units[2], activation=tf.square)
        self.layer4 = bnn.SquareDense(n_units[2], n_units[3], activation=tf.square)
        self.layer5 = bnn.SquareDense(n_units[3], n_units[4], activation=tf.square)
        self.layer6 = bnn.SquareDense(n_units[4], 1)

    def call(self, inputs):
        x = self.bihomogeneous(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = tf.math.log(x)
        return x

class OuterProductNN_k2(keras.Model):
   
    def __init__(self):
        super(OuterProductNN_k2, self).__init__()
        self.bihomogeneous_k2 = bnn.Bihomogeneous_k2()
        self.layer1 = bnn.WidthOneDense(15**2, 1)

    def call(self, inputs):
        x = self.bihomogeneous_k2(inputs)
        x = self.layer1(x)
        x = tf.math.log(x)
        return x


class OuterProductNN_k3(keras.Model):
   
    def __init__(self):
        super(OuterProductNN_k3, self).__init__()
        self.bihomogeneous_k3 = bnn.Bihomogeneous_k3()
        self.layer1 = bnn.WidthOneDense(35**2, 1)

    def call(self, inputs):
        x = self.bihomogeneous_k3(inputs)
        x = self.layer1(x)
        x = tf.math.log(x)
        return x

class OuterProductNN_k4(keras.Model):
   
    def __init__(self):
        super(OuterProductNN_k4, self).__init__()
        self.bihomogeneous_k4 = bnn.Bihomogeneous_k4()
        self.layer1 = bnn.WidthOneDense(70**2, 1)

    def call(self, inputs):
        with tf.device('/cpu:0'):
            x = self.bihomogeneous_k4(inputs)
        with tf.device('/gpu:0'):
            x = self.layer1(x)
            x = tf.math.log(x)
        return x

class k2_twolayers(keras.Model):

    def __init__(self, n_units):
        super(k2_twolayers, self).__init__()
        self.bihomogeneous_k2 = bnn.Bihomogeneous_k2()
        self.layer1 = bnn.SquareDense(15**2, n_units[0], activation=tf.square)
        self.layer2 = bnn.SquareDense(n_units[0], n_units[1], activation=tf.square)
        self.layer3 = bnn.SquareDense(n_units[1], 1)

    def call(self, inputs):
        x = self.bihomogeneous_k2(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = tf.math.log(x)
        return x


class k2_threelayers(keras.Model):

    def __init__(self, n_units):
        super(k2_threelayers, self).__init__()
        self.bihomogeneous_k2 = bnn.Bihomogeneous_k2()
        self.layer1 = bnn.SquareDense(15**2, n_units[0], activation=tf.square)
        self.layer2 = bnn.SquareDense(n_units[0], n_units[1], activation=tf.square)
        self.layer3 = bnn.SquareDense(n_units[1], n_units[2], activation=tf.square)
        self.layer4 = bnn.SquareDense(n_units[2], 1)

    def call(self, inputs):
        x = self.bihomogeneous_k2(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = tf.math.log(x)
        return x

class k4_onelayer(keras.Model):

    def __init__(self, n_units):
        super(k4_onelayer, self).__init__()
        self.bihomogeneous_k4 = bnn.Bihomogeneous_k4()
        self.layer1 = bnn.SquareDense(70**2, n_units[0], activation=tf.square)
        self.layer2 = bnn.SquareDense(n_units[0], 1)

    def call(self, inputs):
        x = self.bihomogeneous_k4(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = tf.math.log(x)
        return x

class k4_twolayers(keras.Model):

    def __init__(self, n_units):
        super(k4_twolayers, self).__init__()
        self.bihomogeneous_k4 = bnn.Bihomogeneous_k4()
        self.layer1 = bnn.SquareDense(70**2, n_units[0], activation=tf.square)
        self.layer2 = bnn.SquareDense(n_units[0], n_units[1], activation=tf.square)
        self.layer3 = bnn.SquareDense(n_units[1], 1)

    def call(self, inputs):
        x = self.bihomogeneous_k4(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = tf.math.log(x)
        return x

