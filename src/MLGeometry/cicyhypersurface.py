import logging
import numpy as np
import sympy as sp
import mpmath
from multiprocessing import Pool
from .hypersurface import Hypersurface
from .hypersurface import RealHypersurface

__all__ = ['RealHypersurface', 'CICYRealHypersurface']

class CICYHypersurface(Hypersurface):
    """Complete Intersection Calabi-Yau Hypersurface"""

    def solve_points(self, n_trios):
        # Optimized solve_points
        ztrios = self.generate_random_projective(n_trios, 3) # (n_trios, 3, n_dim) complex list
        
        # Prepare polynomial coefficients for the intersection
        coeff_a = sp.var('a0:{}'.format(self.n_dim))
        coeff_b = sp.var('b0:{}'.format(self.n_dim))
        coeff_c = sp.var('c0:{}'.format(self.n_dim))
        sp.var('t0:2')
        
        coeff_zip = zip(coeff_a, coeff_b, coeff_c)
        plane = [t0*a + t1*b + c for (a, b, c) in coeff_zip]
        
        # Substitute plane equation into functions
        # self.function is a list of expressions (Matrix)
        poly_t = sp.Matrix(self.function).subs([(self.coordinates[i], plane[i])
                                                 for i in range(self.n_dim)])

        # Extract coefficients w.r.t (t0, t1)
        coeffs_list = [sp.Poly(poly, (t0, t1)).coeffs() for poly in poly_t]
        monoms_list = [sp.Poly(poly, (t0, t1)).monoms() for poly in poly_t]

        # Lambdify coefficient generation
        # Input: all a, b, c. Output: list of coefficient lists.
        all_coeffs = coeff_a + coeff_b + coeff_c
        coeffs_func = sp.lambdify(all_coeffs, coeffs_list, "numpy")

        # Prepare inputs for multiprocessing
        # ztrio is (3, n_dim). Flatten to (3*n_dim,).
        # We need to map coeffs_func over ztrios
        ztrios_flat = [np.array(zt).flatten() for zt in ztrios]
        
        # Evaluate coefficients
        coeffs_evaluated = [coeffs_func(*zt) for zt in ztrios_flat]

        # Solve in parallel
        # monoms_list is constant structure
        monoms_list_repeated = [monoms_list] * n_trios
        
        points = self.solve_points_multiprocessing(coeffs_evaluated, monoms_list_repeated, ztrios)
        return points

    def solve_points_multiprocessing(self, coeffs_list, monoms_list, ztrios):
        points = []
        with Pool() as pool:
            for point in pool.starmap(CICYHypersurface.solve_poly, zip(coeffs_list, monoms_list, ztrios)):
                if point is not None:
                    points.append(point)
        return points

    @staticmethod
    def solve_poly(coeff_list, monom_list, ztrio):
        point = None
        
        # Define function to solve: f_i(t0, t1) = 0
        def func_t(t0, t1):
            res = []
            for coeffs, monoms in zip(coeff_list, monom_list):
                val = 0
                for c, m in zip(coeffs, monoms):
                    val += c * (t0**m[0]) * (t1**m[1])
                res.append(val)
            return res

        # Try to find root
        # CICY usually codimension k, embedded in CP^N.
        # Line intersection gives 0-dimensional set of points on the linear slice?
        # No, ztrio defines a Plane (2D complex).
        # Intersection of Plane (2D) with CICY (dim N-k) in CP^N?
        # If N=5, k=2 (CICY 3-fold). Plane is CP^2. Intersection is points?
        # Yes, standard method finds points.
        
        try:
            # Initial guess
            t_init = np.random.randn(2).tolist() # Changed from original to be more explicit.
            t_solved = mpmath.findroot(func_t, t_init)
            
            # Reconstruct point in CP^N
            # z = t0*z0 + t1*z1 + z2 (where z0,z1,z2 are the trio vectors)
            t_array = np.array(t_solved.tolist(), dtype=np.complex64)
            # Add 1.0 for the constant term corresponding to z2
            t_weights = np.array([t_array[0], t_array[1], 1.0], dtype=np.complex64)
            
            # ztrio is (3, n_dim)
            point = np.dot(t_weights, np.array(ztrio))
            return point.tolist()
        except:
            return None

    def get_grad(self):
        func = sp.Matrix(self.function)
        grad = func.jacobian(self.affine_coordinates)
        return grad

    def get_hol_n_form(self, coord):
        try:
            # For CICY, we delete columns corresponding to 'coord' (which is a list?)
            # Wait, parent expects coord to be int.
            # CICY 'max_grad_coordinate' is likely a list/tuple of indices?
            # autopatch logic below suggests `max_grad_coord` is `[i, j]`.
            # grad is (n_funcs, n_affine).
            # We take determinant of the minor defined by columns `coord`.
            return 1 / self.grad[:, coord].det()
        except:
            logging.exception('Error calculating holomorphic n-form')
            return []

    def autopatch(self):
        # 1. Projective patches (same as parent vectorized)
        # Removed check for self.n_points to avoid uninitialized attribute access.
        # self.points is guaranteed to be set before calling this.
        if len(self.points) == 0:
            return

        norms = np.abs(self.points)
        max_indices = np.argmax(norms, axis=1)
        
        for i in range(self.n_dim):
            mask = (max_indices == i)
            if np.any(mask):
                points_sub = self.points[mask]
                norm_factors = points_sub[:, i][:, np.newaxis]
                points_normalized = points_sub / norm_factors
                self.set_patch(points_normalized, norm_coord=i)

        # Remove empty patches
        self.patches = [p for p in self.patches if p.n_points > 0]

        # 2. Subpatches
        for patch in self.patches:
            # Construct Jacobian Determinant Matrix
            # grad is (n_eq, n_vars).
            # We want determinants of all (n_eq x n_eq) minors.
            # n_vars = n_dim - 1 (affine).
            # n_eq = len(self.function).
            
            n_affine = len(patch.affine_coordinates)
            n_eq = len(self.function)
            
            # Generate all pairs/tuples of columns
            import itertools
            # We need to choose n_eq columns from n_affine columns
            col_indices = list(itertools.combinations(range(n_affine), n_eq))
            
            # Build list of determinant expressions
            det_exprs = []
            for cols in col_indices:
                sub_matrix = patch.grad[:, cols]
                det_exprs.append(sub_matrix.det())
                
            # Lambdify the vector of determinants
            # output shape: (n_combinations,)
            det_func = sp.lambdify(self.coordinates, det_exprs, 'numpy')
            
            # Evaluate on all points
            # patch.points: (N, n_dim)
            dets_vals = np.array(det_func(*patch.points.T)).T # (N, n_combinations)
            
            dets_norms = np.abs(dets_vals)
            best_det_indices = np.argmax(dets_norms, axis=1) # (N,) index into col_indices
            
            # Assign points to subpatches
            for idx, cols in enumerate(col_indices):
                mask = (best_det_indices == idx)
                if np.any(mask):
                    points_sub = patch.points[mask]
                    # max_grad_coord is the tuple of columns chosen
                    patch.set_patch(points_sub, norm_coord=patch.norm_coordinate, max_grad_coord=list(cols))

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
        
        # Solve locally? Or just substitute?
        # "subs({coord: func ...})" implies we are using the defining equations 
        # to replace the ignored coordinates.
        # But this requires solving the system, or assuming 'func' is the definition.
        # In `cicyhypersurface.py`, `self.function` is a list of expressions = 0.
        # The logic in original code:
        # local_coordinates = sp.Matrix(W).subs({coord: func for coord, func in zip(ignored_coordinate, self.function)})
        # This replaces w_i with f_j. This calculates the map from embedding to "function space"?
        # Actually, it seems to define the local coords as (w_k, f_j) effectively?
        # And then Jacobians...
        
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

    def solve_points_multiprocessing(self, coeffs_list, monoms_list, ztrios):
        points = []
        with Pool() as pool:
            for point in pool.starmap(RealCICYHypersurface.solve_poly, zip(coeffs_list, monoms_list, ztrios)):
                if point is not None:
                    points.append(point)
        return points

    @staticmethod
    def solve_poly(coeff_list, monom_list, ztrio):
        # Same as CICY solve_poly but returns only if real?
        # Parent RealHypersurface.solve_poly_real logic was: solve, check imag part < 1e-8.
        
        # Here we solve for t.
        # We need t such that point is real? Or just t is real?
        # If ztrio is real, and t is real, point is real.
        # RealCICYHypersurface likely implies real manifold.
        
        def func_t(t0, t1):
            res = []
            for coeffs, monoms in zip(coeff_list, monom_list):
                val = 0
                for c, m in zip(coeffs, monoms):
                    val += c * (t0**m[0]) * (t1**m[1])
                res.append(val)
            return res

        try:
            t_init = np.random.randn(2).tolist() # Real guess
            t_solved = mpmath.findroot(func_t, t_init)
            
            # Check if solution is real
            # mpmath returns mpc usually.
            if abs(t_solved[0].imag) > 1e-8 or abs(t_solved[1].imag) > 1e-8:
                return None
                
            t_array = np.array([t_solved[0].real, t_solved[1].real, 1.0], dtype=np.float64)
            
            # ztrio should be real if generate_random_projective works
            point = np.dot(t_array, np.array(ztrio))
            return point.tolist()
        except:
            return None
