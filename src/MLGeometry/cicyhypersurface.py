"""Defines a Python class for complete intersection hypersurfaces"""

import itertools
import logging

import numpy as np
import sympy as sp
from .hypersurface import Hypersurface
from .hypersurface import RealHypersurface
from . import config

__all__ = ['CICYHypersurface', 'RealCICYHypersurface', 'batch_system_roots']

# Number of trios that solve_points handles at one time. A chunk holds arrays of
# shape (chunk, d1*d2+1, d1+d2, d1+d2), so this constant bounds the peak memory.
POINTS_CHUNK_TRIOS = 5000


class CICYHypersurface(Hypersurface):
    """Complete Intersection Calabi-Yau Hypersurface"""

    def __init__(self, *args, **kwargs):
        # solve_points runs inside the parent constructor, so the cache for the
        # plane coefficients has to exist before that call.
        self._plane_coeff_func = None
        super().__init__(*args, **kwargs)

    def solve_points(self, n_trios, chunk_size=None):
        """Generates random points on the intersection with Monte Carlo.

        Each trio of random points spans a plane in CP^N, and that plane meets
        the two hypersurfaces in d1*d2 points, so the result holds d1*d2 *
        n_trios points. The normalization fixes the projective scale, so that the
        two equations determine exactly two remaining coordinates, and Newton
        steps solve them to the round-off level.
        """
        if len(self.function) != 2:
            raise ValueError('solve_points needs exactly two defining equations,'
                             ' because the resultant eliminates one parameter')
        if chunk_size is None:
            chunk_size = POINTS_CHUNK_TRIOS

        if self._plane_coeff_func is None:
            self._plane_coeff_func = self.get_plane_coeff(lambdify=True)

        # Solve one chunk of trios at a time, so that the intermediate arrays
        # stay small even when n_trios is large.
        chunks = []
        n_solved = 0
        while n_solved < n_trios:
            ztrios = self.generate_random_projective(
                min(chunk_size, n_trios - n_solved), 3)
            a = ztrios[:, 0]
            b = ztrios[:, 1]
            c = ztrios[:, 2]

            grids = self.eval_plane_coeff(a, b, c)
            t0, t1 = batch_system_roots(*grids)

            points, norm_coords = self.normalize_points(
                self.get_plane_points(t0, t1, a, b, c))
            chunks.append(self.refine_points(points, norm_coords))
            n_solved += len(ztrios)

        return np.concatenate(chunks)

    def get_plane_coeff(self, lambdify=False):
        """Coefficients of f(t0*a + t1*b + c) as a polynomial in t0 and t1.

        A trio of points (a, b, c) spans the plane t0*a + t1*b + c in CP^N. Each
        equation gives a dense grid of coefficients, where entry (m, n) belongs
        to t0^m * t1^n and the entries above the total degree are zero.
        """
        coeff_a = [sp.symbols('a'+str(i)) for i in range(self.n_dim)]
        coeff_b = [sp.symbols('b'+str(i)) for i in range(self.n_dim)]
        coeff_c = [sp.symbols('c'+str(i)) for i in range(self.n_dim)]
        t0, t1 = sp.symbols('t0 t1')

        # Substitute plane equation z = t0*a + t1*b + c into f(z) = 0
        plane = [t0*x + t1*y + w for (x, y, w) in zip(coeff_a, coeff_b, coeff_c)]
        substitution = {self.coordinates[i]: plane[i] for i in range(self.n_dim)}

        grids = []
        for function in self.function:
            degree = sp.Poly(function, *self.coordinates).total_degree()
            poly = sp.Poly(sp.expand(function.subs(substitution,
                                                   simultaneous=True)), t0, t1)
            grid = [[sp.Integer(0)] * (degree + 1) for _ in range(degree + 1)]
            for (m, n), coeff in zip(poly.monoms(), poly.coeffs()):
                grid[m][n] = coeff
            grids.append([grid[m][n] for m in range(degree + 1)
                          for n in range(degree + 1)])

        if lambdify:
            return [sp.lambdify([coeff_a, coeff_b, coeff_c], grid, 'numpy')
                    for grid in grids]
        return grids

    def eval_plane_coeff(self, a, b, c):
        """Coefficient grids of every plane in a chunk, one call per equation."""
        grids = []
        for function, coeff_func in zip(self.function, self._plane_coeff_func):
            degree = sp.Poly(function, *self.coordinates).total_degree()
            # A coefficient that does not depend on a, b or c comes back as a
            # scalar, so broadcast it.
            flat = coeff_func(list(a.T), list(b.T), list(c.T))
            flat = np.stack(np.broadcast_arrays(*flat), axis=-1)
            grids.append(flat.reshape(len(a), degree + 1, degree + 1))
        return grids

    def get_plane_points(self, t0, t1, a, b, c):
        """Points where each plane meets the intersection, as a flat list.

        Each of the d1*d2 common roots of the two equations gives one point.
        """
        points = (t0[:, :, np.newaxis] * a[:, np.newaxis, :]
                  + t1[:, :, np.newaxis] * b[:, np.newaxis, :]
                  + c[:, np.newaxis, :])
        return points.reshape(-1, self.n_dim)

    def max_grad_columns(self, points, norm_coords):
        """Columns of the largest Jacobian minor in each point.

        The two equations determine these two coordinates from the others, so
        refine_points moves them. norm_coords names the normalized column, which
        no minor may use, and it takes one index for every point or one for all.
        """
        n_eq = len(self.function)
        combos = np.array(list(itertools.combinations(range(self.n_dim), n_eq)))

        jac = self._eval_jac(self.get_newton_funcs()[1], points)
        # minors[point, combination, equation, column]
        minors = np.moveaxis(jac[:, :, combos], 2, 1)
        dets = np.abs(np.linalg.det(minors))

        # A minor that uses the normalized column is not allowed
        forbidden = np.any(combos[np.newaxis] == np.reshape(norm_coords, (-1, 1, 1)),
                           axis=-1)
        dets = np.where(forbidden, -1.0, dets)

        return combos[np.argmax(dets, axis=1)]

    def refine_points(self, points, norm_coords, n_steps=2):
        """Newton steps that solve the two equations in two coordinates.

        The normalized coordinate keeps the exact value one, and the two
        equations determine exactly two other coordinates. Two steps bring the
        residual of both equations down to the round-off level.
        """
        function_func, jac_func = self.get_newton_funcs()
        columns = self.max_grad_columns(points, norm_coords)

        for _ in range(n_steps):
            values = np.stack(np.broadcast_arrays(
                *function_func(*points.T)), axis=1)
            jac = self._eval_jac(jac_func, points)
            minors = np.take_along_axis(jac, columns[:, np.newaxis, :], axis=2)

            step = np.linalg.solve(minors, values[..., np.newaxis])[..., 0]
            np.put_along_axis(points, columns,
                              np.take_along_axis(points, columns, 1) - step, 1)

        return points

    def get_newton_funcs(self):
        """Numerical equations and their Jacobian, for refine_points."""
        if self._newton_funcs is None:
            functions = list(self.function)
            jacobian = [[f.diff(coord) for coord in self.coordinates]
                        for f in functions]
            self._newton_funcs = (
                sp.lambdify(self.coordinates, functions, 'numpy'),
                sp.lambdify(self.coordinates, jacobian, 'numpy'))
        return self._newton_funcs

    @staticmethod
    def _eval_jac(jac_func, points):
        """Jacobian at every point, as an array of shape (N, n_eq, n_dim).

        An entry that is constant comes back as a scalar.
        """
        rows = jac_func(*points.T)
        return np.stack([np.stack(np.broadcast_arrays(*row), axis=1)
                         for row in rows], axis=1)

    def get_grad(self):
        func = sp.Matrix(self.function)
        grad = func.jacobian(self.affine_coordinates)
        return grad

    def get_hol_n_form(self, coord):
        try:
            return 1 / self.grad[:, coord].det()
        except Exception:
            logging.exception('Error calculating holomorphic n-form')
            return []

    def autopatch(self):
        """Assigns points to patches based on the coordinate with largest magnitude."""
        if len(self.points) == 0:
            return

        max_indices = np.argmax(np.abs(self.points), axis=1)

        for i in range(self.n_dim):
            mask = (max_indices == i)
            if np.any(mask):
                # Every point here has its largest modulus at i, so this divides
                # by z_i and makes that coordinate exactly one.
                points_normalized, _ = self.normalize_points(self.points[mask])

                self.set_patch(points_normalized, norm_coord=i)

        # Remove empty patches
        self.patches = [p for p in self.patches if p.n_points > 0]

        # Subpatches, chosen by the largest Jacobian minor
        for patch in self.patches:
            # set_patch indexes the affine coordinates, which are the
            # coordinates without the normalized one, so drop that column.
            columns = patch.max_grad_columns(patch.points, patch.norm_coordinate)
            max_grad_indices = columns - (columns > patch.norm_coordinate)

            n_eq = len(self.function)
            n_affine = len(patch.affine_coordinates)
            for cols in itertools.combinations(range(n_affine), n_eq):
                mask = np.all(max_grad_indices == np.array(cols), axis=1)
                if np.any(mask):
                    patch.set_patch(patch.points[mask],
                                    norm_coord=patch.norm_coordinate,
                                    max_grad_coord=list(cols))

    def set_patch(self, points_on_patch, norm_coord=None, max_grad_coord=None):
        new_patch = CICYHypersurface(self.coordinates,
                                     self.function,
                                     points=points_on_patch,
                                     norm_coordinate=norm_coord,
                                     max_grad_coordinate=max_grad_coord)
        self.patches.append(new_patch)

    def get_restriction(self, ignored_coord=None, lambdify=False):
        if ignored_coord is None:
            ignored_coord = self.max_grad_coordinate

        # ignored_coord is a list of indices in affine coordinates

        # Prepare symbolic substitution
        W = sp.var('w0:{}'.format(len(self.affine_coordinates)))

        # Identify variables to be eliminated
        ignored_vars = [np.array(W)[i] for i in ignored_coord]

        # Preserving original logic:
        # Create map {w_i: f_i} for i in ignored_indices.
        # Note: len(ignored_coord) should equal len(self.function).

        subs_dict = {np.array(W)[idx]: expr for idx, expr in zip(ignored_coord, self.function)}

        # Construct local coordinates vector (symbolic)
        # Start with W, replace selected Ws with corresponding functions
        local_coordinates = sp.Matrix(W).subs(subs_dict)

        # Now substitute back W -> actual affine coordinates
        w_to_z = {w: z for w, z in zip(W, self.affine_coordinates)}
        local_coordinates = local_coordinates.subs(w_to_z)

        # Jacobian of this map w.r.t affine coordinates
        jac = local_coordinates.jacobian(self.affine_coordinates)
        restriction = jac.inv()

        # Delete columns corresponding to the eliminated coordinates
        # Note: we need to delete multiple columns.
        # Deleting shifts indices, so delete in reverse order of indices.
        sorted_ignored = sorted(ignored_coord, reverse=True)
        for idx in sorted_ignored:
            restriction.col_del(idx)

        if lambdify is True:
            return sp.lambdify(self.coordinates, restriction.tolist(), 'numpy')
        return restriction


class RealCICYHypersurface(CICYHypersurface, RealHypersurface):

    def generate_random_projective(self, n_set, n_pt_in_a_set):
        # Use RealHypersurface's generator (real points)
        return RealHypersurface.generate_random_projective(self, n_set, n_pt_in_a_set)

    def get_plane_points(self, t0, t1, a, b, c, tol=1e-8):
        """Points where each plane meets the intersection, in RP^N.

        A complex root gives no point in RP^N, so the number of points per plane
        is not fixed. a, b and c are real, so the real part of the combination is
        the point that the real parts of t0 and t1 give.
        """
        points = (t0[:, :, np.newaxis] * a[:, np.newaxis, :]
                  + t1[:, :, np.newaxis] * b[:, np.newaxis, :]
                  + c[:, np.newaxis, :])
        keep = (np.abs(t0.imag) < tol) & (np.abs(t1.imag) < tol)
        return points.real[keep] + 0j


# Helper functions
def sylvester_matrices(grid_1, grid_2, samples):
    """Sylvester matrices of the two equations in t1, at the given t0 samples.

    grid_i has shape (P, d_i+1, d_i+1) and holds the coefficient of t0^m * t1^n
    at entry (m, n). The result has shape (P, K, d1+d2, d1+d2).
    """
    d1 = grid_1.shape[-1] - 1
    d2 = grid_2.shape[-1] - 1

    # coefficient of t1^n at each sample of t0, in descending order of n
    powers_1 = samples[:, np.newaxis] ** np.arange(d1 + 1)
    powers_2 = samples[:, np.newaxis] ** np.arange(d2 + 1)
    row_1 = np.einsum('k m, p m n -> p k n', powers_1, grid_1)[..., ::-1]
    row_2 = np.einsum('k m, p m n -> p k n', powers_2, grid_2)[..., ::-1]

    size = d1 + d2
    matrices = np.zeros(grid_1.shape[:1] + (len(samples), size, size),
                        dtype=np.result_type(grid_1, grid_2, samples))
    for i in range(d2):
        matrices[:, :, i, i:i + d1 + 1] = row_1
    for i in range(d1):
        matrices[:, :, d2 + i, i:i + d2 + 1] = row_2
    return matrices


def batch_system_roots(grid_1, grid_2, n_steps=2):
    """All d1*d2 common roots (t0, t1) of two polynomials, for a batch of planes.

    The Sylvester resultant in t1 is a polynomial in t0 of degree d1*d2. Samples
    at the roots of unity turn its recovery into a discrete Fourier transform,
    which avoids an ill-conditioned Vandermonde solve.
    """
    d1 = grid_1.shape[-1] - 1
    d2 = grid_2.shape[-1] - 1
    n_planes = grid_1.shape[0]
    degree = d1 * d2
    n_samples = degree + 1

    samples = np.exp(2j * np.pi * np.arange(n_samples) / n_samples)
    det = np.linalg.det(sylvester_matrices(grid_1, grid_2, samples))
    # det[k] = sum_j g_j w^(j k), so a forward transform returns n_samples * g_j
    coeff = np.fft.fft(det, axis=-1) / n_samples

    t0 = _companion_roots(coeff[:, ::-1])

    # f1(t0, .) is univariate in t1. Take its roots and keep the one that also
    # makes f2 vanish.
    grid_1_rep = np.broadcast_to(grid_1[:, np.newaxis],
                                 (n_planes, degree) + grid_1.shape[1:])
    grid_2_rep = np.broadcast_to(grid_2[:, np.newaxis],
                                 (n_planes, degree) + grid_2.shape[1:])
    coeff_t1 = np.einsum('p d m, p d m n -> p d n',
                         t0[..., np.newaxis] ** np.arange(d1 + 1), grid_1_rep)
    candidates = _companion_roots(coeff_t1[..., ::-1])

    t0_wide = np.broadcast_to(t0[..., np.newaxis], candidates.shape)
    grid_2_wide = np.broadcast_to(grid_2[:, np.newaxis, np.newaxis],
                                  (n_planes, degree, d1) + grid_2.shape[1:])
    best = np.argmin(np.abs(_eval_grid(grid_2_wide, t0_wide, candidates)), axis=-1)
    t1 = np.take_along_axis(candidates, best[..., np.newaxis], axis=-1)[..., 0]

    # Newton on the pair, which removes the round-off of the resultant
    for _ in range(n_steps):
        values = np.stack([_eval_grid(grid_1_rep, t0, t1),
                           _eval_grid(grid_2_rep, t0, t1)], axis=-1)
        jac = np.stack([
            np.stack([_eval_grid(_d_dt0(grid_1_rep), t0, t1),
                      _eval_grid(_d_dt1(grid_1_rep), t0, t1)], axis=-1),
            np.stack([_eval_grid(_d_dt0(grid_2_rep), t0, t1),
                      _eval_grid(_d_dt1(grid_2_rep), t0, t1)], axis=-1),
        ], axis=-2)
        step = np.linalg.solve(jac, values[..., np.newaxis])[..., 0]
        t0 = t0 - step[..., 0]
        t1 = t1 - step[..., 1]

    return t0, t1


def _companion_roots(coeff):
    """Roots from coefficients in descending order, over the leading axes."""
    degree = coeff.shape[-1] - 1
    monic = coeff[..., ::-1] / coeff[..., 0, np.newaxis]

    companion = np.zeros(coeff.shape[:-1] + (degree, degree), dtype=coeff.dtype)
    companion[..., 1:, :-1] = np.identity(degree - 1)
    companion[..., :, -1] = -monic[..., :degree]

    return np.linalg.eigvals(companion)


def _eval_grid(grid, t0, t1):
    """Value of the polynomial that a coefficient grid holds."""
    powers_0 = t0[..., np.newaxis] ** np.arange(grid.shape[-2])
    powers_1 = t1[..., np.newaxis] ** np.arange(grid.shape[-1])
    return np.einsum('... m, ... n, ... m n -> ...', powers_0, powers_1, grid)


def _d_dt0(grid):
    """Coefficient grid of the derivative with respect to t0."""
    return np.roll(grid * np.arange(grid.shape[-2])[:, np.newaxis], -1, axis=-2)


def _d_dt1(grid):
    """Coefficient grid of the derivative with respect to t1."""
    return np.roll(grid * np.arange(grid.shape[-1]), -1, axis=-1)
