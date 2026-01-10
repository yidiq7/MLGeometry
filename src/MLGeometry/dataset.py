"""
Data loading and processing utilities.
"""

import numpy as np
import jax.numpy as jnp
from . import config

__all__ = ['generate_dataset', 'dataset_on_patch', 'to_jax']


def generate_dataset(patch, jax: bool = True) -> dict:
    """
    Generates a consolidated dataset from a Hypersurface object and its patches.
    
    Args:
        patch: A Hypersurface object (root).
        jax: If True (default), converts the dataset to JAX arrays before returning.
        
    Returns:
        A dictionary containing concatenated arrays (NumPy or JAX).
    """
    # Collect all leaf patches that contain points
    leaf_patches = _collect_leaf_patches(patch)
    
    if not leaf_patches:
        raise ValueError("No patches with points found in the hypersurface.")

    # Process each patch to extract tensors
    datasets = []
    for p in leaf_patches:
        if p.n_points > 0:
            datasets.append(dataset_on_patch(p))
            
    if not datasets:
        raise ValueError("Patches exist but contain no points.")

    # Concatenate all datasets efficiently
    merged_dataset = {}
    keys = datasets[0].keys()
    
    for key in keys:
        merged_dataset[key] = np.concatenate([d[key] for d in datasets], axis=0)
        
    if jax:
        return to_jax(merged_dataset)
    return merged_dataset


def to_jax(dataset: dict) -> dict:
    """
    Converts a dataset dictionary of NumPy arrays to JAX arrays with appropriate types.
    Use this before passing data to JAX training loops to move data to device.
    
    Args:
        dataset: Dictionary containing numpy arrays.
        
    Returns:
        Dictionary containing JAX arrays.
    """
    return {
        'points': jnp.array(dataset['points'], dtype=config.complex_dtype),
        'Omega_Omegabar': jnp.array(jnp.real(dataset['Omega_Omegabar']), dtype=config.real_dtype),
        'mass': jnp.array(jnp.real(dataset['mass']), dtype=config.real_dtype),
        'restriction': jnp.array(dataset['restriction'], dtype=config.complex_dtype)
    }


def _collect_leaf_patches(patch) -> list:
    """Recursively collects patches that do not have subpatches."""
    if not patch.patches:
        return [patch]
    
    leaves = []
    for subpatch in patch.patches:
        leaves.extend(_collect_leaf_patches(subpatch))
    return leaves


def dataset_on_patch(patch) -> dict:
    """
    Computes numerical tensors for a single patch.
    
    This invokes the JAX-based vectorized methods on the patch object
    to generate sections, restrictions, and volume forms.
    """
    # 1. Initialize numerical quantities
    # The '1' indicates k=1 (bundle power) for Fubini-Study mass calculation
    patch.s_jax_1, patch.J_jax_1 = patch.num_s_J_jax(k=1)
    
    # Initialize restriction matrix (needed for FS volume form calculation)
    patch.r_jax = patch.num_restriction_jax()
    
    # 2. Extract Points
    x = np.array(patch.points, dtype=config.np_complex_dtype)
    
    # 3. Compute Omega_Omegabar (Target Measure)
    # Returns float32 array
    y = np.array(np.real(patch.num_Omega_Omegabar_jax()), dtype=config.np_real_dtype)
    
    # 4. Compute Mass (Reference Measure)
    # Fubini-Study volume form for identity metric
    # This uses patch.r_jax internally
    fs_vol = np.array(np.real(patch.num_FS_volume_form_jax('identity', k=1)), dtype=config.np_real_dtype)
    
    # Mass reweighting factor
    mass = y / fs_vol

    # 5. Compute Full Restriction Matrix for the Model
    # The model works in embedding coordinates (d_amb).
    # We need to pull back the metric from embedding to manifold.
    # restriction_affine (d_man, d_aff) @ trans_tensor (d_aff, d_amb)
    
    # Construct projection matrix P: (n_dim-1, n_dim)
    # Removes the row corresponding to the norm coordinate (which is fixed to 1)
    trans_mat = np.delete(np.identity(patch.n_dim), patch.norm_coordinate, axis=0)
    trans_tensor = jnp.array(trans_mat, dtype=config.complex_dtype)
    
    # patch.r_jax: (batch, n_manifold, n_affine)
    # trans_tensor: (n_affine, n_ambient)
    # Result: (batch, n_manifold, n_ambient)
    
    restriction = jnp.matmul(patch.r_jax, trans_tensor)
    restriction = np.array(restriction, dtype=config.np_complex_dtype)

    dataset = {
        'points': x,
        'Omega_Omegabar': y,
        'mass': mass,
        'restriction': restriction
    }

    return dataset
