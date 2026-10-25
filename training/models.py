import math
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from MLGeometry import bihomoNN as bnn
from MLGeometry import config
from typing import Any, Callable, Optional, Sequence, Tuple

__all__ = ['Kahler_potential', 'zerolayer',
           'OuterProductNN_k2','OuterProductNN_k3','OuterProductNN_k4',
           'k2_twolayers', 'k2_threelayers','k4_onelayer','k4_twolayers',
           'SpectralNetwork', 'SpectralLinearNetwork', 'SpectralSirenNetwork',
           'ScaledSineDense', 'SpectralWangLaiNetwork',
           'PolynomialBranch', 'PolynomialCorrection',
           'calibrate_polynomial', 'reduced_potential_mean',
           'recurrence_features', 'UnitRecurrence', 'calibrate_recurrence', 'recurrence_gram_error']

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

class SpectralLinearNetwork(nn.Module):
    """SpectralNetwork with the linear output amp * (h @ w), with no sinh and no bias.

    A bias of the output is a constant of the potential, so it does not change the metric.
    The output layer has the name 'output', so that a solve can find its kernel w. The
    kernel starts from a normal distribution with the standard deviation out_std.
    """
    layers: Sequence[int]
    amp: Any = 1.0
    activation: Callable = jnp.tanh
    out_std: float = 1e-3
    d: int = 5

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = bnn.Spectral(d=self.d)(inputs)
        for feat in self.layers:
            x = bnn.Dense(features=feat, activation=self.activation)(x)
        x = bnn.Dense(features=1, use_bias=False, name='output',
                      kernel_init=nn.initializers.normal(stddev=self.out_std))(x)
        return self.amp * x

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

def recurrence_features(z: jnp.ndarray, phi_prev: Optional[Callable] = None, k0: int = 1,
                        phi_shift: float = 0.0) -> jnp.ndarray:
    """The d^2 real features of degree 0 of UnitRecurrence at the points z.

    They are the real parts of z_a z_b_bar for a <= b and the imaginary parts for a < b. The
    imaginary parts of the diagonal are zero, so they are left out. With phi_prev = None the
    weight is 1 / |z|^2, the Fubini-Study weight. With phi_prev, the frozen potential of the
    earlier stages in the class k0 H, the weight is
    exp(-(phi_prev - k0 log|z|^2 - phi_shift) / k0) / |z|^2, which is exp(-phi_prev / k0) up to
    the constant factor exp(phi_shift / k0): the base weight.

    Args:
        z: Complex array of shape (..., d).
        phi_prev: Optional frozen potential, a function from (..., d) to (...).
        k0: The class of phi_prev.
        phi_shift: Constant that keeps the base weight near 1, from reduced_potential_mean.

    Returns:
        Real array of shape (..., d^2).
    """
    d = z.shape[-1]
    zz = z[..., :, None] * jnp.conj(z[..., None, :])
    rows, cols = np.triu_indices(d)
    off = rows != cols
    x = jnp.concatenate([jnp.real(zz[..., rows, cols]), jnp.imag(zz[..., rows[off], cols[off]])], axis=-1)
    norm_sq = jnp.sum(jnp.real(z * jnp.conj(z)), axis=-1)
    if phi_prev is None:
        weight = 1.0 / norm_sq
    else:
        weight = jnp.exp(-(phi_prev(z) - k0 * jnp.log(norm_sq) - phi_shift) / k0) / norm_sq
    return x * weight[..., None]

class UnitRecurrence(nn.Module):
    """The units of a learned recurrence of Vandermonde with Arnoldi type, and psi = sum_m c_m u_m.

    The features x are those of recurrence_features. With u_0 = 1, level l + 1 forms the
    products t = (x A_l^T) * (u_l B_l^T) and the units u_(l+1) = (t - U_l H_l) R_l^-1, with U_l
    all units of the levels 0 to l. A_l (widths[l] x n_features), B_l (widths[l] x width of u_l)
    and c are parameters. H_l and R_l^-1 are the fixed collection 'orth'. calibrate_recurrence
    computes them, so that the units are orthonormal on the fit points in the mass-weighted
    inner product. A unit of level l is a polynomial of degree l in x, so psi is a sum over the
    degrees L of P_L(z) times the weight of the features to the power L, with P_L
    bihomogeneous of degree (L, L). psi uses the units of the levels 1 to len(widths).
    """
    widths: Sequence[int]
    n_features: int = 25
    phi_prev: Optional[Callable] = None
    k0: int = 1
    phi_shift: float = 0.0

    def setup(self):
        sizes = [1] + [int(k) for k in self.widths]
        n_before = np.cumsum(sizes)
        levels = range(len(self.widths))
        self.a = [self.param(f'A_{l}', nn.initializers.normal(stddev=1.0 / math.sqrt(self.n_features)),
                             (sizes[l + 1], self.n_features), config.real_dtype) for l in levels]
        self.b = [self.param(f'B_{l}', nn.initializers.normal(stddev=1.0 / math.sqrt(sizes[l])),
                             (sizes[l + 1], sizes[l]), config.real_dtype) for l in levels]
        self.c = self.param('c', nn.initializers.zeros, (int(sum(sizes[1:])),), config.real_dtype)
        self.h = [self.variable('orth', f'H_{l}', jnp.zeros, (int(n_before[l]), sizes[l + 1]),
                                config.real_dtype) for l in levels]
        self.rinv = [self.variable('orth', f'Rinv_{l}', lambda n, dtype: jnp.eye(n, dtype=dtype),
                                   sizes[l + 1], config.real_dtype) for l in levels]

    def all_units(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """All units of the levels 0 to L at the points inputs, an array (..., 1 + sum(widths))."""
        x = recurrence_features(inputs, self.phi_prev, self.k0, self.phi_shift)
        units = jnp.ones(x.shape[:-1] + (1,), x.dtype)
        last = units
        for a, b, h, rinv in zip(self.a, self.b, self.h, self.rinv):
            t = (x @ a.T) * (last @ b.T)
            last = (t - units @ h.value) @ rinv.value
            units = jnp.concatenate([units, last], axis=-1)
        return units

    def units(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """The units of the levels 1 to L at the points inputs, an array (..., sum(widths))."""
        return self.all_units(inputs)[..., 1:]

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return (self.units(inputs) @ self.c)[..., None]

def _pivoted_cholesky(gram: np.ndarray, norms: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy pivoted Cholesky factor of a Gram matrix.

    At each step the column with the largest remaining part relative to its norm enters, while
    this relative part exceeds tol.

    Returns:
        (the indices of the entered columns in their order, the upper triangular R with
        gram[sel][:, sel] = R^T R).
    """
    n = gram.shape[0]
    d = np.diag(gram).astype(np.float64).copy()
    scale = np.where(norms > 0, norms, 1.0)**2
    perm = np.arange(n)
    r = np.zeros((n, n))
    k = 0
    while k < n:
        rel = d[perm[k:]] / scale[perm[k:]]
        j = k + int(np.argmax(rel))
        if not rel[j - k] > tol**2:
            break
        perm[[k, j]] = perm[[j, k]]
        r[:k, [k, j]] = r[:k, [j, k]]
        r[k, k] = np.sqrt(d[perm[k]])
        rest = perm[k + 1:]
        r[k, k + 1:] = (gram[perm[k], rest] - r[:k, k] @ r[:k, k + 1:]) / r[k, k]
        d[rest] -= r[k, k + 1:]**2
        k += 1
    return perm[:k], r[:k, :k]

def calibrate_recurrence(x: jnp.ndarray, weights: jnp.ndarray, width: int, degree: int,
                         key: Optional[Any] = None, a_list: Optional[Sequence] = None,
                         b_list: Optional[Sequence] = None, tol: float = 1e-6) -> dict:
    """Orthonormalizes the products of UnitRecurrence level by level on the fit points.

    Each level forms the products t = (x A^T) * (u B^T) of the last units u, takes out the
    earlier units in two passes of block Gram-Schmidt, and orthonormalizes the rest with a
    pivoted Cholesky factor of its Gram matrix and a second pass. With random matrices
    (a_list = None), A and B are Gaussian with the variance 1 / fan-in, and a product enters
    only if its new part exceeds tol times its norm, so a level can have fewer than `width`
    units. With a_list and b_list, all products enter in their order.

    Args:
        x: Features at the fit points, an array (N, n_features).
        weights: Mass weights of the fit points with the sum 1, an array (N,).
        width: Number of products of each level of the random matrices.
        degree: Number of levels.
        key: PRNG key of the random matrices.
        a_list, b_list: Optional matrices A_l and B_l of all levels.
        tol: Smallest new part, relative to the norm of the product, that enters.

    Returns:
        A dict with 'widths' (the number of units of each level), 'variables' (the collections
        'params' and 'orth' of UnitRecurrence, with c = 0) and 'new_parts' (for each level, the
        new part of each product that entered, relative to its norm, in the order of entry).
    """
    wcol = weights[:, None]
    n_features = x.shape[1]
    units = jnp.ones((x.shape[0], 1), x.dtype)
    last = units
    params, orth, widths, new_parts = {}, {}, [], []
    for l in range(degree):
        if a_list is None:
            key_a, key_b = jax.random.split(jax.random.fold_in(key, l))
            a = jax.random.normal(key_a, (width, n_features), x.dtype) / math.sqrt(n_features)
            b = jax.random.normal(key_b, (width, last.shape[1]), x.dtype) / math.sqrt(last.shape[1])
        else:
            a, b = jnp.asarray(a_list[l]), jnp.asarray(b_list[l])
        t = (x @ a.T) * (last @ b.T)
        norms = jnp.sqrt(jnp.sum(wcol * t**2, axis=0))
        h1 = units.T @ (wcol * t)
        t = t - units @ h1
        h2 = units.T @ (wcol * t)
        t = t - units @ h2
        gram = np.asarray(t.T @ (wcol * t))
        if a_list is None:
            sel, r1 = _pivoted_cholesky(gram, np.asarray(norms), tol)
        else:
            sel = np.arange(t.shape[1])
            r1 = np.linalg.cholesky(gram).T
        r1 = jnp.asarray(r1)
        # q = t_sel R1^-1, then a second pass against the earlier units and within the level.
        q = jax.scipy.linalg.solve_triangular(r1, t[:, sel].T, trans='T', lower=False).T
        h3 = units.T @ (wcol * q)
        q = q - units @ h3
        r2 = jnp.linalg.cholesky(q.T @ (wcol * q)).T
        q = jax.scipy.linalg.solve_triangular(r2, q.T, trans='T', lower=False).T
        # The units at any point: (t_sel - U H) R^-1 with H = (h1 + h2)_sel + h3 R1 and R = R2 R1.
        r = r2 @ r1
        params[f'A_{l}'], params[f'B_{l}'] = a[sel], b[sel]
        orth[f'H_{l}'] = (h1 + h2)[:, sel] + h3 @ r1
        orth[f'Rinv_{l}'] = jax.scipy.linalg.solve_triangular(r, jnp.eye(r.shape[0], dtype=r.dtype), lower=False)
        widths.append(int(len(sel)))
        new_parts.append(np.asarray(jnp.diag(r1)) / np.asarray(norms)[sel])
        units = jnp.concatenate([units, q], axis=1)
        last = q
    params['c'] = jnp.zeros(units.shape[1] - 1, x.dtype)
    return {'widths': widths, 'variables': {'params': params, 'orth': orth}, 'new_parts': new_parts}

def recurrence_gram_error(model: UnitRecurrence, variables: Any, points: jnp.ndarray, mass: jnp.ndarray,
                          batch_size: int = 10000) -> float:
    """max abs(U^T W U - I) of all units of the levels 0 to L on the points, with W = mass / sum(mass)."""
    weights = mass / jnp.sum(mass)

    @jax.jit
    def gram_of(variables, z, w):
        u = model.apply(variables, z, method=UnitRecurrence.all_units)
        return u.T @ (w[:, None] * u)

    gram = 0.0
    for i in range(0, points.shape[0], batch_size):
        gram = gram + gram_of(variables, points[i:i + batch_size], weights[i:i + batch_size])
    return float(jnp.max(jnp.abs(gram - jnp.eye(gram.shape[0], dtype=gram.dtype))))

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