"""
Loss functions for Calabi-Yau metric learning.
"""

import jax.numpy as jnp

__all__ = ['weighted_MAPE', 'weighted_MSE', 'max_error', 'MAPE_plus_max_error']


def weighted_MAPE(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Weighted Mean Absolute Percentage Error."""
    weights = mass / jnp.sum(mass)
    return jnp.sum(jnp.abs(y_true - y_pred) / y_true * weights)


def weighted_MSE(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Weighted Mean Squared Error (of the ratio - 1)."""
    weights = mass / jnp.sum(mass)
    return jnp.sum(jnp.square(y_pred / y_true - 1) * weights)


def max_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Maximum relative error."""
    # Note: mass is unused but kept for interface consistency
    return jnp.max(jnp.abs(y_true - y_pred) / y_true)


def MAPE_plus_max_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Combination of MAPE and Max Error."""
    return max_error(y_true, y_pred, mass) + weighted_MAPE(y_true, y_pred, mass)
