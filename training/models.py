from flax import linen as nn
import jax.numpy as jnp
from MLGeometry import bihomoNN as bnn
from typing import Sequence, Any, List

__all__ = ['zerolayer', 'onelayer', 'twolayers', 'threelayers', 'fourlayers', 
           'fivelayers','OuterProductNN_k2','OuterProductNN_k3','OuterProductNN_k4',
           'k2_twolayers', 'k2_threelayers','k4_onelayer','k4_twolayers',
           'Kahler_potential']

# Helper activation
def square_activation(x):
    return x**2

class Kahler_potential(nn.Module):
    # Attributes (Hyperparameters)
    layers: Sequence[int]
    amp: Any = 1.0
    d: int = 5

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = bnn.Bihomogeneous(d=self.d)(inputs)
        
        # Dynamically create layers based on the 'layers' list
        # The list 'layers' contains hidden layer sizes. 
        # The final density reduction to 1 feature is handled after the loop.
        for feat in self.layers:
            x = bnn.SquareDense(features=feat, activation=square_activation)(x)
            
        # Final layer
        x = bnn.SquareDense(features=1, activation=None)(x)
        
        return self.amp * jnp.log(x)

class zerolayer(nn.Module):
    n_units: Sequence[int] # Not used but for consistency with interface

    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous()(inputs)
        x = bnn.WidthOneDense(features=1)(x)
        return jnp.log(x)

class onelayer(nn.Module):
    n_units: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous()(inputs)
        x = bnn.SquareDense(features=self.n_units[0], activation=square_activation)(x)
        x = bnn.SquareDense(features=1, activation=None)(x)
        return jnp.log(x)

class twolayers(nn.Module):
    n_units: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous()(inputs)
        x = bnn.SquareDense(features=self.n_units[0], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[1], activation=square_activation)(x)
        x = bnn.SquareDense(features=1, activation=None)(x)
        return jnp.log(x)

class threelayers(nn.Module):
    n_units: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous()(inputs)
        x = bnn.SquareDense(features=self.n_units[0], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[1], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[2], activation=square_activation)(x)
        x = bnn.SquareDense(features=1, activation=None)(x)
        return jnp.log(x)

class fourlayers(nn.Module):
    n_units: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous()(inputs)
        x = bnn.SquareDense(features=self.n_units[0], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[1], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[2], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[3], activation=square_activation)(x)
        x = bnn.SquareDense(features=1, activation=None)(x)
        return jnp.log(x)

class fivelayers(nn.Module):
    n_units: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous()(inputs)
        x = bnn.SquareDense(features=self.n_units[0], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[1], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[2], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[3], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[4], activation=square_activation)(x)
        x = bnn.SquareDense(features=1, activation=None)(x)
        return jnp.log(x)

class OuterProductNN_k2(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous_k2()(inputs)
        x = bnn.WidthOneDense(features=1)(x)
        return jnp.log(x)

class OuterProductNN_k3(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous_k3()(inputs)
        x = bnn.WidthOneDense(features=1)(x)
        return jnp.log(x)

class OuterProductNN_k4(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous_k4()(inputs)
        x = bnn.WidthOneDense(features=1)(x)
        return jnp.log(x)

class k2_twolayers(nn.Module):
    n_units: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous_k2()(inputs)
        x = bnn.SquareDense(features=self.n_units[0], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[1], activation=square_activation)(x)
        x = bnn.SquareDense(features=1, activation=None)(x)
        return jnp.log(x)

class k2_threelayers(nn.Module):
    n_units: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous_k2()(inputs)
        x = bnn.SquareDense(features=self.n_units[0], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[1], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[2], activation=square_activation)(x)
        x = bnn.SquareDense(features=1, activation=None)(x)
        return jnp.log(x)

class k4_onelayer(nn.Module):
    n_units: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous_k4()(inputs)
        x = bnn.SquareDense(features=self.n_units[0], activation=square_activation)(x)
        x = bnn.SquareDense(features=1, activation=None)(x)
        return jnp.log(x)

class k4_twolayers(nn.Module):
    n_units: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous_k4()(inputs)
        x = bnn.SquareDense(features=self.n_units[0], activation=square_activation)(x)
        x = bnn.SquareDense(features=self.n_units[1], activation=square_activation)(x)
        x = bnn.SquareDense(features=1, activation=None)(x)
        return jnp.log(x)