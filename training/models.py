from flax import linen as nn
import jax.numpy as jnp
from MLGeometry import bihomoNN as bnn
from MLGeometry import config
from typing import Any, Callable, Optional, Sequence

__all__ = ['Kahler_potential', 'zerolayer',
           'OuterProductNN_k2','OuterProductNN_k3','OuterProductNN_k4',
           'k2_twolayers', 'k2_threelayers','k4_onelayer','k4_twolayers',
           'SpectralNetwork', 'SpectralSirenNetwork',
           'ScaledSineDense', 'SpectralWangLaiNetwork',
           'PolynomialBranch', 'PolynomialCorrection',
           'calibrate_polynomial', 'reduced_potential_mean']

# Helper activation
def square_activation(x):
    return x**2

class Kahler_potential(nn.Module):
    """Bihomogeneous network with an arbitrary number of hidden layers.

    Supersedes the fixed-depth onelayer...fivelayers classes: `layers` is the
    list of hidden widths, followed by an implicit width-1 output layer.
    Kahler_potential(layers=n_units[:hidden]) reproduces those exactly.
    """
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

class SpectralNetwork(nn.Module):
    """Correction potential of one stage on the spectral features z_i z_j_bar / |z|^2.

    `layers` is the list of hidden widths. Each hidden layer is bnn.Dense with Xavier
    initialization and the activation. The width-1 output x gives amp * sinh(x), so amp
    sets the scale of the correction. If out_std is set, the output kernel starts from a
    normal distribution with this standard deviation, so that the stage starts near the
    potential before it. Otherwise it has Xavier initialization.
    """
    layers: Sequence[int]
    amp: Any = 1.0
    activation: Callable = jnp.tanh
    out_std: Optional[float] = None
    d: int = 5

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = bnn.Spectral(d=self.d)(inputs)
        for feat in self.layers:
            x = bnn.Dense(features=feat, activation=self.activation)(x)
        if self.out_std is None:
            x = bnn.Dense(features=1)(x)
        else:
            x = bnn.Dense(features=1, kernel_init=nn.initializers.normal(stddev=self.out_std))(x)
        return self.amp * jnp.sinh(x)

class SpectralSirenNetwork(nn.Module):
    """SpectralNetwork with SIREN hidden layers, bnn.SirenDense with the sine activation.

    The first hidden layer has the SIREN initialization of a first layer. The width-1
    output x of bnn.Dense gives amp * sinh(x).
    """
    layers: Sequence[int]
    amp: Any = 1.0
    omega_0: float = 30.0
    d: int = 5

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = bnn.Spectral(d=self.d)(inputs)
        for i, feat in enumerate(self.layers):
            x = bnn.SirenDense(features=feat, omega_0=self.omega_0, is_first_layer=(i == 0))(x)
        x = bnn.Dense(features=1)(x)
        return self.amp * jnp.sinh(x)

class ScaledSineDense(nn.Module):
    """A dense layer with the output sin(kappa * (x @ W) + b).

    W has the Xavier normal initialization of bnn.Dense, and b starts at zero. Only W gets
    the fixed scale kappa, so kappa sets the frequencies of the units at initialization.
    """
    features: int
    kappa: float = 1.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            inputs: Real array of shape (..., n_in).

        Returns:
            Real array of shape (..., features).
        """
        kernel = self.param('kernel', nn.initializers.xavier_normal(),
                            (inputs.shape[-1], self.features), config.real_dtype)
        bias = self.param('bias', nn.initializers.zeros, (self.features,), config.real_dtype)
        return jnp.sin(self.kappa * jnp.dot(inputs, kernel) + bias)

class SpectralWangLaiNetwork(nn.Module):
    """Correction potential of one stage with the scaled sine first layer of Wang and Lai.

    Reference: Y. Wang and C.-Y. Lai, arXiv:2307.08934 (2023). The first hidden layer is
    ScaledSineDense on the spectral features z_i z_j_bar / |z|^2. The other entries of
    `layers` are bnn.Dense layers with tanh. The output is amp * (h @ w), linear and with no
    bias. The kernel w starts from a normal distribution with the standard deviation out_std.
    """
    layers: Sequence[int]
    kappa: float = 1.0
    amp: Any = 1.0
    out_std: float = 1e-3
    d: int = 5

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = bnn.Spectral(d=self.d)(inputs)
        x = ScaledSineDense(features=self.layers[0], kappa=self.kappa)(x)
        for feat in self.layers[1:]:
            x = bnn.Dense(features=feat, activation=jnp.tanh)(x)
        x = bnn.Dense(features=1, use_bias=False,
                      kernel_init=nn.initializers.normal(stddev=self.out_std))(x)
        return self.amp * x

class PolynomialBranch(nn.Module):
    """A homogeneous polynomial of degree 2^len(widths) in its input.

    Each hidden layer is bnn.SquareDense, h -> (h W)^2, with no bias, divided by a fixed
    scale. The output layer is linear, with signed weights and no bias. Its kernel starts
    from a normal distribution with the standard deviation out_std.
    """
    widths: Sequence[int]
    scales: Optional[Sequence[float]] = None
    out_std: float = 1e-3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, feat in enumerate(self.widths):
            x = bnn.SquareDense(features=feat)(x)
            # A fixed scale keeps the units near 1; it does not change the degree.
            if self.scales is not None:
                x = x / self.scales[i]
        return bnn.Dense(features=1, use_bias=False,
                         kernel_init=nn.initializers.normal(stddev=self.out_std))(x)

class PolynomialCorrection(nn.Module):
    """Correction potential psi = amp * sum_L P_L(x) w_L(z) on x = z_i z_j_bar / |z|^2.

    Each entry of `branches` is the list of widths of one PolynomialBranch P_L, of degree
    L = 2^len(widths) in x. So P_L(x) = N_L(z) / |z|^(2L), with N_L of degree (L, L).
    With phi_prev = None, w_L = 1: this is the Fubini-Study weight. With phi_prev, the
    frozen potential of the earlier stages in the class k0 H, w_L is the base weight

        w_L = exp(-(L / k0) (phi_prev(z) - k0 log|z|^2 - phi_shift)),

    which has degree 0. Then psi = sum_L N_L(z) exp(-(L / k0) phi_prev(z)) up to a constant
    factor of each branch. phi_shift (a constant) keeps the weights near 1. Autodiff goes
    through phi_prev, so the metric of psi includes the derivatives of the weights.
    """
    branches: Sequence[Sequence[int]]
    amp: Any = 1.0
    out_std: float = 1e-3
    scales: Optional[Sequence[Sequence[float]]] = None
    phi_prev: Optional[Callable] = None
    k0: int = 1
    phi_shift: float = 0.0
    d: int = 5

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = bnn.Spectral(d=self.d)(inputs)
        if self.phi_prev is not None:
            log_norm = jnp.log(jnp.sum(jnp.real(inputs * jnp.conj(inputs)), axis=-1))
            reduced = self.phi_prev(inputs) - self.k0 * log_norm - self.phi_shift
        total = 0.0
        for i, widths in enumerate(self.branches):
            scales = None if self.scales is None else self.scales[i]
            p = PolynomialBranch(widths=widths, scales=scales, out_std=self.out_std)(x)[..., 0]
            if self.phi_prev is not None:
                degree = 2 ** len(widths)
                p = p * jnp.exp(-(degree / self.k0) * reduced)
            total = total + p
        return self.amp * total[..., None]

def calibrate_polynomial(model: PolynomialCorrection, params: Any, points: jnp.ndarray) -> list:
    """Fixed scales that give each x^2 layer of each branch an RMS of one on the points.

    The output of a layer is homogeneous in the scales of the layers before it. So one pass
    with no scales gives all of them: s_l = r_l / r_(l-1)^2, with r_l the RMS of the raw
    output of layer l and r_0 = 1.

    Returns:
        (scales, raw_rms): two lists with one list of floats for each branch.
    """
    raw = model.clone(scales=None)
    _, state = raw.apply(params, points, capture_intermediates=True, mutable=['intermediates'])
    inter = state['intermediates']
    scales, raw_rms = [], []
    for i, widths in enumerate(model.branches):
        branch = inter[f'PolynomialBranch_{i}']
        r_prev, s, r_list = 1.0, [], []
        for l in range(len(widths)):
            h = branch[f'SquareDense_{l}']['__call__'][0]
            r = float(jnp.sqrt(jnp.mean(h**2)))
            s.append(r / r_prev**2)
            r_list.append(r)
            r_prev = r
        scales.append(s)
        raw_rms.append(r_list)
    return scales, raw_rms

def reduced_potential_mean(phi_prev: Callable, k0: int, points: jnp.ndarray) -> float:
    """Mean of phi_prev(z) - k0 log|z|^2 on the points, the phi_shift of PolynomialCorrection."""
    log_norm = jnp.log(jnp.sum(jnp.real(points * jnp.conj(points)), axis=-1))
    return float(jnp.mean(phi_prev(points) - k0 * log_norm))

class zerolayer(nn.Module):
    n_units: Sequence[int] # Not used but for consistency with interface

    @nn.compact
    def __call__(self, inputs):
        x = bnn.Bihomogeneous()(inputs)
        x = bnn.WidthOneDense(features=1)(x)
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