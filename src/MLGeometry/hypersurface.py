"""Defines a Python class for hypersurfaces"""  

import numpy as np
import sympy as sp
import jax.numpy as jnp
import jax
from . import config

__all__ = ['Hypersurface', 'RealHypersurface', 'batch_poly_roots',
           'diff', 'diff_conjugate']

# Number of pairs that solve_points handles at one time. A chunk holds arrays of
# shape (chunk, degree, degree), so this constant bounds the peak memory.
POINTS_CHUNK_PAIRS = 20000

class Hypersurface():
    r"""A hypersurface or patch defined both symbolically and numerically."""

    def __init__(self, 
                 coordinates, 
                 function,
                 n_pairs=0, 
                 points=None, 
                 norm_coordinate=None,
                 max_grad_coordinate=None):
        
        self.coordinates = np.array(coordinates)
        self.function = function
        self.n_dim = len(self.coordinates)
        self.norm_coordinate = norm_coordinate
        
        if norm_coordinate is not None:
            self.affine_coordinates = np.delete(self.coordinates, norm_coordinate)
        else:
            self.affine_coordinates = self.coordinates
            
        self.max_grad_coordinate = max_grad_coordinate
        self.patches = []

        # Cache for numerical functions
        self._omega_omegabar_func = None
        self._restriction_func = None
        self._line_coeff_func = None
        self._newton_funcs = None

        if points is None:
            # Generate points if not provided
            points_list = self.solve_points(n_pairs)
            self.points = np.array(points_list, dtype=config.np_complex_dtype)
            self.autopatch()
        else:
            # Convert provided points to numpy array
            self.points = np.array(points, dtype=config.np_complex_dtype)
            
        self.n_points = len(self.points)
        self.grad = self.get_grad()

    def solve_points(self, n_pairs, chunk_size=None):
        """Generates random points on the hypersurface with Monte Carlo.

        Each pair of random points spans a line in CP^N, and that line meets the
        hypersurface in `degree` points, so the result holds degree * n_pairs
        points. The normalization fixes the projective scale, so that f = 0
        determines exactly one remaining coordinate, and Newton steps solve it
        to the round-off level.
        """
        if chunk_size is None:
            chunk_size = POINTS_CHUNK_PAIRS

        if self._line_coeff_func is None:
            self._line_coeff_func = self.get_line_coeff(lambdify=True)

        # Solve one chunk of pairs at a time, so that the intermediate arrays
        # stay small even when n_pairs is large.
        chunks = []
        n_solved = 0
        while n_solved < n_pairs:
            zpairs = self.generate_random_projective(
                min(chunk_size, n_pairs - n_solved), 2)
            a = zpairs[:, 0]
            b = zpairs[:, 1]

            # One call covers every line in the chunk. A coefficient that does
            # not depend on a or b comes back as a scalar, so broadcast it.
            coeff = self._line_coeff_func(list(a.T), list(b.T))
            coeff = np.stack(np.broadcast_arrays(*coeff), axis=-1)

            c_solved = batch_poly_roots(coeff)
            points, norm_coords = self.normalize_points(
                self.get_line_points(c_solved, a, b))
            chunks.append(self.refine_points(points, norm_coords))
            n_solved += len(zpairs)

        return np.concatenate(chunks)

    def get_line_coeff(self, lambdify=False):
        """Coefficients of f(c*a + b) as a polynomial in c, in descending order.

        A pair of points (a, b) spans the line c*a + b in CP^N.
        """
        coeff_a = [sp.symbols('a'+str(i)) for i in range(self.n_dim)]
        coeff_b = [sp.symbols('b'+str(i)) for i in range(self.n_dim)]
        c = sp.symbols('c')

        # Substitute line equation z = c*a + b into f(z) = 0
        line = [c*a + b for (a, b) in zip(coeff_a, coeff_b)]
        function_eval = self.function.subs([(self.coordinates[i], line[i]) for i in range(self.n_dim)])

        # all_coeffs keeps a coefficient that is zero, so that every line in a
        # chunk gives the same number of columns.
        coeff_poly = sp.Poly(function_eval, c).all_coeffs()

        if lambdify:
            return sp.lambdify([coeff_a, coeff_b], coeff_poly, 'numpy')
        return coeff_poly

    def get_line_points(self, c_solved, a, b):
        """Points where each line c*a + b meets the hypersurface, as a flat list.

        Each of the `degree` roots of the line polynomial gives one point.
        """
        points = (c_solved[:, :, np.newaxis] * a[:, np.newaxis, :]
                  + b[:, np.newaxis, :])
        return points.reshape(-1, self.n_dim)

    def max_grad_columns(self, points, norm_coords):
        """Column of the largest gradient in each point, as an index into points.

        f = 0 determines this coordinate from the others, so refine_points moves
        it. norm_coords names the normalized column, which is excluded, and it
        takes one index for every point or one index for all of them.
        """
        _, grad_func = self.get_newton_funcs()

        grad = np.abs(self._eval_grad(grad_func, points))
        grad[np.arange(len(points)), norm_coords] = -1.0

        return np.argmax(grad, axis=1)

    def normalize_points(self, points):
        """Scales each point so that its largest-modulus coordinate is one.

        The division leaves a rounding error in that coordinate, whose exact
        value is one, so the assignment restores it.
        """
        norm_coords = np.argmax(np.abs(points), axis=1)
        rows = np.arange(len(points))

        points = points / points[rows, norm_coords][:, np.newaxis]
        points[rows, norm_coords] = 1.0

        return points, norm_coords

    def refine_points(self, points, norm_coords, n_steps=2):
        """Newton steps that solve f = 0 in one coordinate of each point.

        The normalized coordinate keeps the exact value one, and f = 0 is one
        equation, so it determines exactly one other coordinate. Two steps bring
        the residual of f down to the round-off level.
        """
        function_func, grad_func = self.get_newton_funcs()
        rows = np.arange(len(points))
        columns = self.max_grad_columns(points, norm_coords)

        points = points.copy()
        for _ in range(n_steps):
            f_val = np.asarray(function_func(*points.T))
            grad_val = self._eval_grad(grad_func, points)
            points[rows, columns] -= f_val / grad_val[rows, columns]

        return points

    def get_newton_funcs(self):
        """Numerical f and its gradient in every coordinate, for refine_points."""
        if self._newton_funcs is None:
            grad = [self.function.diff(coord) for coord in self.coordinates]
            self._newton_funcs = (
                sp.lambdify(self.coordinates, self.function, 'numpy'),
                sp.lambdify(self.coordinates, grad, 'numpy'))
        return self._newton_funcs

    @staticmethod
    def _eval_grad(grad_func, points):
        """Gradient of f at every point, as an array of shape (N, n_dim).

        A component of the gradient that is constant comes back as a scalar.
        """
        return np.stack(np.broadcast_arrays(*grad_func(*points.T)), axis=1)

    def generate_random_projective(self, n_set, n_pt_in_a_set):
        """Generate sets of points in CP^N"""
        # Shape: (n_set, n_pt_in_a_set, n_dim) complex
        real_part = np.random.normal(0.0, 1.0, (n_set, n_pt_in_a_set, self.n_dim))
        imag_part = np.random.normal(0.0, 1.0, (n_set, n_pt_in_a_set, self.n_dim))
        return real_part + 1j * imag_part

    def autopatch(self):
        """Assigns points to patches based on the coordinate with largest magnitude."""
        # Removed check for self.n_points to avoid uninitialized attribute access.
        # self.points is guaranteed to be set before calling this.
        
        if len(self.points) == 0:
            return

        # Vectorized autopatch for top-level patches
        max_indices = np.argmax(np.abs(self.points), axis=1)

        for i in range(self.n_dim):
            # Select points belonging to patch i
            mask = (max_indices == i)
            if np.any(mask):
                # Every point here has its largest modulus at i, so this divides
                # by z_i and makes that coordinate exactly one.
                points_normalized, _ = self.normalize_points(self.points[mask])

                self.set_patch(points_normalized, norm_coord=i)

        # Subpatches (based on gradient)
        for patch in self.patches:
            if patch.n_points == 0:
                continue
                
            # set_patch indexes the affine coordinates, which are the
            # coordinates without the normalized one, so drop that column.
            columns = patch.max_grad_columns(patch.points, patch.norm_coordinate)
            max_grad_indices = columns - (columns > patch.norm_coordinate)
            
            for i in range(len(patch.affine_coordinates)):
                mask = (max_grad_indices == i)
                if np.any(mask):
                    points_sub = patch.points[mask]
                    patch.set_patch(points_sub, norm_coord=patch.norm_coordinate, max_grad_coord=i)

    def set_patch(self, points_on_patch, norm_coord=None, max_grad_coord=None):
        new_patch = Hypersurface(self.coordinates, 
                                 self.function, 
                                 points=points_on_patch, 
                                 norm_coordinate=norm_coord,
                                 max_grad_coordinate=max_grad_coord)
        self.patches.append(new_patch)

    def list_patches(self):
        print("Number of Patches:", len(self.patches))
        for i, patch in enumerate(self.patches):
            print(f"Points on patch {i+1} : {patch.n_points}")

    def get_grad(self):
        return [self.function.diff(coord) for coord in self.affine_coordinates]

    def get_hol_n_form(self, coord):
        if coord is not None:
            return 1 / self.grad[coord]
        return [self.get_hol_n_form(i) for i in range(len(self.affine_coordinates))]

    def get_omega_omegabar(self, lambdify=False):
        if self.patches == [] and self.max_grad_coordinate is not None:
            hol_n_form = self.get_hol_n_form(self.max_grad_coordinate)
            expr = hol_n_form * sp.conjugate(hol_n_form)
            if lambdify:
                return sp.lambdify(self.coordinates, expr, 'numpy')
            return expr
        
        # Recursive case is handled by dataset generation usually, 
        # but if called on top level, return list
        return [p.get_omega_omegabar(lambdify) for p in self.patches]

    def get_sections(self, k, lambdify=False):
        t = sp.symbols('t')
        GenSec = sp.prod(1/(1-(t*zz)) for zz in self.coordinates)
        poly = sp.series(GenSec, t, n=k+1).coeff(t**k)
        
        sections = []
        while poly != 0:
            lt = sp.LT(poly)
            sections.append(lt)
            poly = poly - lt
            
        if lambdify:
            return sp.lambdify(self.coordinates, sections, 'numpy'), len(sections)
        return np.array(sections), len(sections)

    def get_restriction(self, ignored_coord=None, lambdify=False):
        if ignored_coord is None:
            ignored_coord = self.max_grad_coordinate
            
        ignored_val = self.affine_coordinates[ignored_coord]
        
        # Symbolic substitution to eliminate ignored_coord
        # Jacobian(local) w.r.t (affine)
        
        # Strategy:
        # local_coords = affine_coords with ignored_val replaced by -f(...)/... ?
        # Actually, we are restricting the tangent bundle.
        # This part assumes we can solve f=0 for ignored_val locally.
        
        # The symbolic logic:
        local_coordinates = sp.Matrix(self.affine_coordinates).subs(ignored_val, self.function)
        affine_coordinates = sp.Matrix(self.affine_coordinates)
        
        jac = local_coordinates.jacobian(affine_coordinates)
        restriction = jac.inv()
        restriction.col_del(ignored_coord)
        
        if lambdify:
            return sp.lambdify(self.coordinates, restriction.tolist(), 'numpy')
        return restriction

    # Numerical Methods with JAX/Vectorization

    def num_s_J_jax(self):
        """Vectorized degree-one sections and their Jacobian.

        The sections are the homogeneous coordinates themselves, so the Jacobian
        with respect to the affine coordinates is the constant projection that
        drops the row of the normalized coordinate.
        """
        s_jax = self.points[:, np.newaxis, :] # (N, 1, n_dim)

        J_mat = np.delete(np.identity(self.n_dim), self.norm_coordinate, 0) # (n_dim-1, n_dim)
        J_jax = np.tile(J_mat, (self.n_points, 1, 1)) # (N, n_dim-1, n_dim)

        return jnp.array(s_jax), jnp.array(J_jax)

    def num_Omega_Omegabar_jax(self):
        """Vectorized Omega Omegabar"""
        if self._omega_omegabar_func is None:
            self._omega_omegabar_func = self.get_omega_omegabar(lambdify=True)
            
        # Func takes coordinates
        vals = self._omega_omegabar_func(*self.points.T)
        return jnp.array(jnp.real(vals), dtype=config.real_dtype)

    def num_restriction_jax(self):
        """Vectorized restriction matrix"""
        if self._restriction_func is None:
            self._restriction_func = self.get_restriction(lambdify=True)
            
        r_raw = self._restriction_func(*self.points.T)
        
        # Robust handling of inhomogeneous output (scalars mixed with arrays)
        if isinstance(r_raw, (list, tuple)):
            # Flatten list of lists
            rows = len(r_raw)
            cols = len(r_raw[0]) if rows > 0 else 0
            flat = [elem for row in r_raw for elem in row]
            
            # Broadcast all elements to shape (N,)
            flat_broad = np.broadcast_arrays(*flat)
            
            # Stack to (rows*cols, N) then reshape to (rows, cols, N)
            r_vals = np.stack(flat_broad).reshape(rows, cols, -1)
            
            # Move axis to get (N, rows, cols)
            r_vals = np.moveaxis(r_vals, -1, 0)
        else:
            # Already an array (rows, cols, N)
            r_vals = np.array(r_raw)
            r_vals = np.moveaxis(r_vals, -1, 0)

        # Original code did r.append(res.T). So we want (N, cols, rows)
        r_jax = np.swapaxes(r_vals, 1, 2)
        
        return jnp.array(r_jax, dtype=config.complex_dtype)

    def num_FS_volume_form_jax(self):
        """Fubini-Study volume form restricted to the hypersurface.

        Requires r_jax to have been set from num_restriction_jax.
        """
        FS_metric = self.num_FS_metric_jax()
        r_jax = self.r_jax # (N, n_manifold, n_affine)

        r_dag = jnp.swapaxes(jnp.conj(r_jax), -1, -2)
        vol_mat = jnp.matmul(r_jax, jnp.matmul(FS_metric, r_dag))

        det_vol = jnp.linalg.det(vol_mat)
        return jnp.real(det_vol)

    def num_FS_metric_jax(self):
        """Pullback of the Fubini-Study metric to the affine coordinates.

        Requires s_jax_1 and J_jax_1 to have been set from num_s_J_jax. Here
        M = n_dim is the number of degree-one sections and K = n_dim - 1 the
        number of affine coordinates.
        """
        s_jax = self.s_jax_1 # (N, 1, M)
        J_jax = self.J_jax_1 # (N, K, M)

        # G = A/alpha - B/alpha^2, with alpha = s s^H
        s_dag = jnp.swapaxes(jnp.conj(s_jax), -1, -2) # (N, M, 1)
        alpha = jnp.matmul(s_jax, s_dag) # (N, 1, 1)

        J_dag = jnp.swapaxes(jnp.conj(J_jax), -1, -2) # (N, M, K)

        A = jnp.matmul(J_jax, J_dag) # (N, K, K)

        b = jnp.matmul(s_jax, J_dag) # (N, 1, K)
        b_dag = jnp.swapaxes(jnp.conj(b), -1, -2) # (N, K, 1)
        B = jnp.matmul(b_dag, b) # (N, K, K)

        G = A / alpha - B / (alpha**2)
        return G

# Helper functions
def batch_poly_roots(coeff):
    """Roots of a batch of polynomials, given coefficients in descending order.

    The last axis of coeff holds degree + 1 coefficients, and any number of
    leading axes index the batch. The roots of each polynomial are the
    eigenvalues of its companion matrix, and numpy solves the whole stack of
    matrices in compiled code.
    """
    degree = coeff.shape[-1] - 1
    # Ascending coefficients of the monic polynomial
    monic = coeff[..., ::-1] / coeff[..., 0, np.newaxis]

    companion = np.zeros(coeff.shape[:-1] + (degree, degree), dtype=coeff.dtype)
    companion[..., 1:, :-1] = np.identity(degree - 1)
    companion[..., :, -1] = -monic[..., :degree]

    return np.linalg.eigvals(companion)


def diff_conjugate(expr, coordinate):
    coord_bar = sp.symbols('coord_bar')
    expr_diff = expr.subs(sp.conjugate(coordinate), coord_bar).diff(coord_bar)
    expr_diff = expr_diff.subs(coord_bar, sp.conjugate(coordinate))
    return expr_diff

def diff(expr, coordinate):
    coord_bar = sp.symbols('coord_bar')
    expr_diff = expr.subs(sp.conjugate(coordinate), coord_bar).diff(coordinate)
    expr_diff = expr_diff.subs(coord_bar, sp.conjugate(coordinate))
    return expr_diff

class RealHypersurface(Hypersurface):
    """Hypersurface defined over Real numbers."""
    
    def generate_random_projective(self, n_set, n_pt_in_a_set):
        """Generate sets of real points in RP^N (embedded in CP^N)."""
        real_part = np.random.normal(0.0, 1.0, (n_set, n_pt_in_a_set, self.n_dim))
        return real_part + 0j

    def get_line_points(self, c_solved, a, b, tol=1e-8):
        """Points where each line c*a + b meets the hypersurface, in RP^N.

        A complex root of the line polynomial gives no point in RP^N, so the
        number of points per line is not fixed. A root that passes the test keeps
        a small imaginary part. a and b are real, so the real part of c*a + b is
        the point that the real part of c gives.
        """
        points = (c_solved[:, :, np.newaxis] * a[:, np.newaxis, :]
                  + b[:, np.newaxis, :])
        return points.real[np.abs(c_solved.imag) < tol] + 0j
