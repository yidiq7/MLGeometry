import tensorflow as tf
import numpy as np

__all__ = ['generate_dataset', 'dataset_on_patch']

def generate_dataset(HS):
    dataset = None
    for patch in HS.patches:
        for subpatch in patch.patches:
            new_dataset = dataset_on_patch(subpatch)
            if dataset is None:
                dataset = new_dataset
            else:
                dataset = dataset.concatenate(new_dataset)
    return dataset

def dataset_on_patch(patch):

    # So that you don't need to invoke set_k()
    patch.s_tf_1, patch.J_tf_1 = patch.num_s_J_tf(k=1)
    patch.omega_omegabar = patch.get_omega_omegabar(lambdify=True)
    patch.restriction = patch.get_restriction(lambdify=True)
    patch.r_tf = patch.num_restriction_tf()

    x = tf.convert_to_tensor(np.array(patch.points, dtype=np.complex64))
    y = tf.cast(patch.num_Omega_Omegabar_tf(), dtype=tf.float32)

    mass = y / tf.cast(patch.num_FS_volume_form_tf('identity', k=1), dtype=tf.float32)

    # The Kahler metric calculated by complex_hessian will include the derivative of the norm_coordinate, 
    # here we transform the restriction so that the corresponding column and row will be ignored in the hessian
    trans_mat = np.delete(np.identity(patch.degree), patch.norm_coordinate, axis=0)
    trans_tensor = tf.convert_to_tensor(np.array(trans_mat, dtype=np.complex64))
    restriction = tf.matmul(patch.r_tf, trans_tensor) 

    dataset = tf.data.Dataset.from_tensor_slices((x, y, mass, restriction))

    return dataset

