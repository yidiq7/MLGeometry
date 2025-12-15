"""Defines a Python class for hypersurfaces"""  

from multiprocessing import Pool
import mpmath
import numpy as np
import sympy as sp
import jax.numpy as jnp
import jax

__all__ = ['Hypersurface', 'RealHypersurface', 'diff', 'diff_conjugate']

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
        
        if points is None:
            # Generate points if not provided
            points_list = self.solve_points(n_pairs)
            self.points = np.array(points_list, dtype=np.complex64)
            self.autopatch()
        else:
            # Convert provided points to numpy array
            self.points = np.array(points, dtype=np.complex64)
            
        self.n_points = len(self.points)
        self.grad = self.get_grad()
        
        # Cache for numerical functions
        self._omega_omegabar_func = None
        self._restriction_func = None

    def solve_points(self, n_pairs):
        """Generates random points on the hypersurface with Monte Carlo using multiprocessing.""" 
        zpairs = self.generate_random_projective(n_pairs, 2)
        
        # Prepare polynomial coefficients for the line intersection
        coeff_a = [sp.symbols('a'+str(i)) for i in range(self.n_dim)]
        coeff_b = [sp.symbols('b'+str(i)) for i in range(self.n_dim)]
        c = sp.symbols('c')
        
        # Substitute line equation z = c*a + b into f(z) = 0
        line = [c*a + b for (a, b) in zip(coeff_a, coeff_b)]
        function_eval = self.function.subs([(self.coordinates[i], line[i]) for i in range(self.n_dim)])
        
        # Get polynomial coefficients w.r.t 'c'
        poly = sp.Poly(function_eval, c)
        coeff_poly = poly.coeffs()
        
        # Create a fast function to get coefficients given a and b
        get_coeff = sp.lambdify([coeff_a, coeff_b], coeff_poly, 'numpy')

        # Solve in parallel
        points = self.solve_points_multiprocessing(zpairs, get_coeff)
        return points

    def solve_points_multiprocessing(self, zpairs, get_coeff):
        points = []
        with Pool() as pool:
            # Pre-calculate coefficients for all pairs
            # zpairs is list of [pt1, pt2]. pt1 is list of coords.
            # get_coeff takes (a_vec, b_vec)
            coeffs_iter = [get_coeff(zp[0], zp[1]) for zp in zpairs]
            
            # Map solver
            for points_d in pool.starmap(Hypersurface.solve_poly, zip(zpairs, coeffs_iter)):
                points.extend(points_d)
        return points

    def generate_random_projective(self, n_set, n_pt_in_a_set):
        """Generate sets of points in CP^N"""
        # Optimized generation
        # Shape: (n_set, n_pt_in_a_set, n_dim) complex
        real_part = np.random.normal(0.0, 1.0, (n_set, n_pt_in_a_set, self.n_dim))
        imag_part = np.random.normal(0.0, 1.0, (n_set, n_pt_in_a_set, self.n_dim))
        return (real_part + 1j * imag_part).tolist()

    @staticmethod
    def solve_poly(zpair, coeff):
        points_d = []
        try:
            # mpmath is used for high precision root finding
            c_solved = mpmath.polyroots(coeff) 
            a = np.array(zpair[0])
            b = np.array(zpair[1])
            for pram_c in c_solved:
                # Reconstruct point: z = c*a + b
                pt = complex(pram_c) * a + b
                points_d.append(pt.tolist())
        except Exception:
            pass
        return points_d
    
    def autopatch(self):
        """Assigns points to patches based on the coordinate with largest magnitude."""
        # Removed check for self.n_points to avoid uninitialized attribute access.
        # self.points is guaranteed to be set before calling this.
        
        if len(self.points) == 0:
            return

        # Vectorized autopatch for top-level patches
        norms = np.abs(self.points)
        max_indices = np.argmax(norms, axis=1)
        
        for i in range(self.n_dim):
            # Select points belonging to patch i
            mask = (max_indices == i)
            if np.any(mask):
                points_in_patch = self.points[mask]
                # Normalize points: divide by z_i
                # points_in_patch[:, i] is the norm coordinate values
                norm_factors = points_in_patch[:, i][:, np.newaxis]
                points_normalized = points_in_patch / norm_factors
                
                self.set_patch(points_normalized, norm_coord=i)

        # Subpatches (based on gradient)
        for patch in self.patches:
            if patch.n_points == 0:
                continue
                
            # Compute gradients for all points in the patch
            # We need a vectorized gradient function
            if not hasattr(patch, '_grad_func'):
                patch._grad_func = sp.lambdify(self.coordinates, patch.grad, 'numpy')
            
            # Evaluate gradient: output shape (n_dim_affine, n_points) usually, or list of arrays
            # patch.points is (N, n_dim)
            # *patch.points.T unpacks to columns (z0_vec, z1_vec, ...)
            grads = patch._grad_func(*patch.points.T)
            # Handle potential list of mixed scalars/arrays in gradient output
            if isinstance(grads, (list, tuple)):
                broad_grads = np.broadcast_arrays(*grads)
                grads = np.stack(broad_grads, axis=0)
            else:
                grads = np.stack(grads, axis=0)
            
            grad_norms = np.abs(grads) # (n_dim_affine, N)
            max_grad_indices = np.argmax(grad_norms, axis=0) # (N,)
            
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
        
        # Jacobian of embedding map (locally)
        # But this logic seems to rely on 'function' being the replacement rule?
        # Typically self.function is f(z)=0.
        # subs(ignored_val, self.function) replaces z_i with f(z).
        # This is only valid if f(z) ~ z_i locally?
        # Original code used this logic, preserving it for correctness of the method 
        # (assuming it implements a specific math trick or implicit function thm derivative).
        
        jac = local_coordinates.jacobian(affine_coordinates)
        restriction = jac.inv()
        restriction.col_del(ignored_coord)
        
        if lambdify:
            return sp.lambdify(self.coordinates, restriction.tolist(), 'numpy')
        return restriction

    # Numerical Methods with JAX/Vectorization

    def num_s_J_jax(self, k=-1):
        """Vectorized computation of sections and Jacobian."""
        if k == 1:
            # Simple case for k=1 mass formula
            s_jax = self.points[:, np.newaxis, :] # (N, 1, n_dim)
            
            # Jacobian J for k=1 (projective coordinates)
            # effectively identity minus the norm coordinate row
            # shape (n_dim-1, n_dim)
            J_mat = np.delete(np.identity(self.n_dim), self.norm_coordinate, 0) # (n_dim-1, n_dim)
            J_jax = np.tile(J_mat, (self.n_points, 1, 1)) # (N, n_dim-1, n_dim)
            # Note: original code J was (n_dim, n_dim-1) per point? 
            # Original: J = np.delete(I, norm, 0). (n_dim-1, n_dim).
            # Then J_vec.append(J).
            # J_jax = array(J_vec).
            # But wait, later we do J.conj().T
            # Let's check consistency with dataset.py
            
            # In dataset.py: restriction = jnp.matmul(patch.r_jax, trans_tensor)
            # patch.num_s_J_jax is used for Kahler metric.
            
            return jnp.array(s_jax), jnp.array(J_jax)
        
        else:
            # General k
            # self.sections is lambdified function returning list/array of sections
            # evaluate on all points
            
            # sections_func returns (n_sections, N) or (N, n_sections)?
            # lambdify with 'numpy' usually returns shape broadcasted from inputs.
            # inputs are (N,). So outputs (n_sections, N).
            
            s_vals = np.array(self.sections(*self.points.T)).T # (N, n_sections)
            s_jax = s_vals[:, np.newaxis, :] # (N, 1, n_sections)
            
            # Jacobian
            # self.sections_jacobian is lambdified Matrix (n_sections, n_affine)
            # returns (n_sections, n_affine, N) usually? Or list of lists of arrays?
            # SymPy lambdify of Matrix returns array of shape (rows, cols) where elements are arrays if inputs are arrays?
            # Or (rows, cols, N)?
            # It usually returns array of object if not careful, or (rows, cols, ...).
            # We need to check dimensions or reshape.
            
            J_raw = self.sections_jacobian(*self.points.T) 
            # Handle potential inhomogeneous output from lambdify
            if isinstance(J_raw, (list, tuple)):
                rows = len(J_raw)
                cols = len(J_raw[0])
                flat = [elem for row in J_raw for elem in row]
                flat_broad = np.broadcast_arrays(*flat)
                J_vals = np.stack(flat_broad).reshape(rows, cols, -1)
                J_vals = np.moveaxis(J_vals, -1, 0) # (N, rows, cols)
            else:
                J_vals = np.array(J_raw) # (rows, cols, N)
                J_vals = np.moveaxis(J_vals, -1, 0) # (N, rows, cols)
            
            # We need J.T (transpose of matrix, not batch)
            # So (N, n_affine, n_sections)
            J_jax = np.swapaxes(J_vals, 1, 2)
            
            return jnp.array(s_jax), jnp.array(J_jax)

    def num_Omega_Omegabar_jax(self):
        """Vectorized Omega Omegabar"""
        if self._omega_omegabar_func is None:
            self._omega_omegabar_func = self.get_omega_omegabar(lambdify=True)
            
        # Func takes coordinates
        vals = self._omega_omegabar_func(*self.points.T)
        return jnp.array(vals, dtype=jnp.float32)

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
        
        return jnp.array(r_jax, dtype=jnp.complex64)

    def num_FS_volume_form_jax(self, h_matrix, k=-1):
        # Uses JAX for batch computation
        kahler_metric = self.num_kahler_metric_jax(h_matrix, k)
        r_jax = self.r_jax # (N, n_dim-1, n_dim) usually?
        
        # r_jax is (N, N_local, N_embed_affine)
        
        # H @ G @ H_dag
        r_dag = jnp.swapaxes(jnp.conj(r_jax), -1, -2)
        
        # Ensure matrix multiplication is (N, ..., ...)
        vol_mat = jnp.matmul(r_jax, jnp.matmul(kahler_metric, r_dag))
        
        det_vol = jnp.linalg.det(vol_mat)
        return jnp.real(det_vol)

    def num_kahler_metric_jax(self, h_matrix, k=-1):
        if isinstance(h_matrix, str):
            if h_matrix == 'identity':
                dim = self.n_dim if k == 1 else self.n_sections
                h_matrix = np.identity(dim, dtype=np.complex64)
            elif h_matrix == 'FS':
                h_matrix = self.h_FS.astype(np.complex64)

        h_jax = jnp.array(h_matrix, dtype=jnp.complex64) # (M, M)

        if k == 1:
            s_jax = self.s_jax_1
            J_jax = self.J_jax_1
        else:
            s_jax = self.s_jax
            J_jax = self.J_jax    

        # s_jax: (N, 1, M)
        # J_jax: (N, K, M) -- wait.
        # Check num_s_J_jax for k=1:
        # J_jax is (N, n_dim, n_dim-1) -> (N, M, K).
        # Check num_s_J_jax for k!=1:
        # J_jax is (N, n_affine, n_sections) -> (N, K, M) or (N, M, K)?
        # J_raw was (n_sec, n_aff). Transposed to (n_aff, n_sec).
        # So J_jax is (N, n_aff, n_sec). (N, K, M).
        
        # Formula: G = A/alpha - B/alpha^2
        # alpha = s h s^H
        s_dag = jnp.swapaxes(jnp.conj(s_jax), -1, -2) # (N, M, 1)
        alpha = jnp.matmul(s_jax, jnp.matmul(h_jax, s_dag)) # (N, 1, 1)
        
        # A = J h J^H  (using h_matrix constant broadcast)
        # J_jax (N, K, M). h_jax (M, M).
        J_dag = jnp.swapaxes(jnp.conj(J_jax), -1, -2) # (N, M, K)
        
        # h @ J^H -> (N, M, K)
        h_Jdag = jnp.matmul(h_jax, J_dag) 
        
        # J @ h_Jdag -> (N, K, K)
        A = jnp.matmul(J_jax, h_Jdag)
        
        # B = (s h J^H)^H @ (s h J^H)
        # b = s @ h @ J^H -> (N, 1, K)
        b = jnp.matmul(s_jax, h_Jdag)
        b_dag = jnp.swapaxes(jnp.conj(b), -1, -2) # (N, K, 1)
        
        # b_dag @ b -> (N, K, K)
        B = jnp.matmul(b_dag, b)
        
        G = A / alpha - B / (alpha**2)
        return G

# Helper functions
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
        return (real_part + 0j).tolist()

    def solve_points_multiprocessing(self, zpairs, get_coeff):
        points = []
        with Pool() as pool:
            coeffs_iter = [get_coeff(zp[0], zp[1]) for zp in zpairs]
            for points_d in pool.starmap(RealHypersurface.solve_poly_real, zip(zpairs, coeffs_iter)):
                points.extend(points_d)
        return points

    @staticmethod
    def solve_poly_real(zpair, coeff):
        points_d = []
        try:
            c_solved = mpmath.polyroots(coeff) 
            a = np.array(zpair[0])
            b = np.array(zpair[1])
            for pram_c in c_solved:
                if abs(complex(pram_c).imag) < 1e-8:
                    pt = complex(pram_c) * a + b
                    points_d.append(pt.tolist())
        except Exception:
            pass
        return points_d
