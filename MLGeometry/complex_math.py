import tensorflow as tf

def gradients_zbar(func, x):
    dx_real = tf.gradients(tf.math.real(func), x)
    dx_imag = tf.gradients(tf.math.imag(func), x)
    return (dx_real + dx_imag*tf.constant(1j, dtype=x.dtype)) / 2

@tf.autograph.experimental.do_not_convert
def complex_hessian(func, x):
    # Take a real function and calculate dzdzbar(f)
    #grad = gradients_z(func, x)
    grad = tf.math.conj(tf.gradients(func, x))
    hessian = tf.stack([gradients_zbar(tmp[0], x)[0]
                        for tmp in tf.unstack(grad, axis=2)],
                       axis = 1) / 2.0
 
    return hessian 

