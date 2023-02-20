"""Defines a function to calculate the complex hessian

(The following comment is written in markdown)

In Tensorflow, `tf.gradient` with respect to a complex variable is defined as 
$2\frac{\partial Real(f)}{\partial \bar{z}}$ for practical reasons.
To obtain the gradient with the mathematical definition, rewrite 
$\frac{\partial f}{\partial \bar{z}}$ as $\frac{\partial Real(f)}{\partial \bar{z}} +
\frac{\partial Imag(f)}{\partial \bar{z}}$, and calculate each term using `tf.gradients`

To obtain the complex hessian dzdzbar(f), where f is a real function,
first calculate ∂f/∂z by taking conjugate of `tf.gradient`, then take
the second derivative with `gradients_zbar`. By default, Tensorflow can only 
handle the gradients of a list of scalar with respect to anther list of scalar. 
So the list of vectors from the first derivative need to be unstacked to lists of 
scalars, then take the second derivative and stack them back.

A simple test of the result is to calculate the hessian of z0z0bar + z1z1bar:

    @tf.function
    def inner_product(x):
        a = tf.math.real(tf.reduce_sum(x*tf.math.conj(x),1))
        return mlg.complex_math.complex_hessian(a, x)

    foo = tf.convert_to_tensor([[1.0-1.0j, 2.0+2.0j]], dtype=tf.complex64)
    print(inner_product(foo))

The result is indeed as expected

    tf.Tensor(
    [[[1.+0.j 0.+0.j]
     [0.+0.j 1.+0.j]]], shape=(1, 2, 2), dtype=complex64)
"""

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

