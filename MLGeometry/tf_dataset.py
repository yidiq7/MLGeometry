import tensorflow as tf
import numpy as np

__all__ = ['generate_dataset', 'dataset_on_patch']

def generate_dataset(patch):
    dataset = None
    if patch.patches == []:
        dataset = dataset_on_patch(patch) 
    else:
        for subpatch in patch.patches:
            new_dataset = generate_dataset(subpatch) 
            if dataset is None: 
                dataset = new_dataset
            else:
                dataset = dataset.concatenate(new_dataset)   
    return dataset

def dataset_on_patch(patch):

    # To calculate the numerical tensors, one needs to invoke function set_k() first
    # to lambdify the sympy expression to generate the python functions used for different k.
    # However the full set of set_k() is too slow for large k. Here the minimum 
    # required functions are invoked so that one does not need to invoke set_k().
    patch.s_tf_1, patch.J_tf_1 = patch.num_s_J_tf(k=1)
    patch.omega_omegabar = patch.get_omega_omegabar(lambdify=True)
    patch.restriction = patch.get_restriction(lambdify=True)
    patch.r_tf = patch.num_restriction_tf()

    x = tf.convert_to_tensor(np.array(patch.points, dtype=np.complex64))
    y = tf.cast(patch.num_Omega_Omegabar_tf(), dtype=tf.float32)

    mass = y / tf.cast(patch.num_FS_volume_form_tf('identity', k=1), dtype=tf.float32)

    # The Kahler metric calculated by complex_hessian includes the derivative of 
    # the norm_coordinate. Here the restriction is linear transformed so that 
    # the corresponding column and row will be ignored in the hessian.
    trans_mat = np.delete(np.identity(patch.n_dim), patch.norm_coordinate, axis=0)
    trans_tensor = tf.convert_to_tensor(np.array(trans_mat, dtype=np.complex64))
    restriction = tf.matmul(patch.r_tf, trans_tensor) 

    FS_metric = patch.num_kahler_metric_tf('FS')
    dataset = tf.data.Dataset.from_tensor_slices((x, y, mass, restriction, FS_metric))

    return dataset

