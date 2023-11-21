import logging
import numpy as np
import sympy as sp
#import tensorflow as tf
import mpmath
from multiprocessing import Pool
from .hypersurface import Hypersurface
from .hypersurface import RealHypersurface

__all__ = ['RealHypersurface', 'CICYRealHypersurface']

class CICYHypersurface(Hypersurface):

    def solve_points(self, n_trios):
        points = []
        ztrios = self.generate_random_projective(n_trios, 3)
        coeff_a = sp.var('a0:{}'.format(self.n_dim))
        coeff_b = sp.var('b0:{}'.format(self.n_dim))
        coeff_c = sp.var('c0:{}'.format(self.n_dim))
        sp.var('t0:2')
        coeff_zip = zip(coeff_a, coeff_b, coeff_c)
        plane = [t0*a + t1*b + c for (a, b, c) in coeff_zip]
        # Add another function & poly & coeffs here
        poly_t = sp.Matrix(self.function).subs([(self.coordinates[i], plane[i])
                                                 for i in range(self.n_dim)])

        coeffs_list = [sp.Poly(poly,(t0, t1)).coeffs() for poly in poly_t]
        monoms_list = [sp.Poly(poly,(t0, t1)).monoms() for poly in poly_t]

        coeffs_func = sp.lambdify([coeff_a + coeff_b + coeff_c], coeffs_list, "numpy")

        coeffs_list = []
        for ztrio in ztrios:
            coeffs_list.append(coeffs_func(np.array(ztrio).flatten()))

        monoms_list = [monoms_list] * n_trios

        points = self.solve_points_multiprocessing(coeffs_list, monoms_list, ztrios)

        return points

    def solve_points_multiprocessing(self, coeffs_list, monoms_list, ztrios):
        points = []
        with Pool() as pool:
            for point in pool.starmap(CICYHypersurface.solve_poly, zip(coeffs_list, monoms_list, ztrios)):
                points.append(point)

        points = list(filter(lambda x: x is not None, points))
        return points

    @staticmethod
    def solve_poly(coeff_list, monom_list, ztrio):
        point = None

        def func_t(t0, t1):
            return [sum([coeff * t0**monom[0]*t1**monom[1] for coeff, monom in zip(coeffs, monoms)]) for coeffs, monoms in zip(coeff_list, monom_list)]

        for attempt in range(1):
            try:
                t_real = np.random.randn(4)
                t_init = [complex(t_real[0], t_real[1]), complex(t_real[2], t_real[3])]
                #t_init = np.random.randn(2).tolist()
                t_solved = mpmath.findroot(func_t, t_init)
                t_array = np.array(t_solved.tolist(), dtype=np.complex64)
                t_array = np.concatenate((t_array, np.array([[1.0+0.0j]])))
                point = np.add.reduce(t_array * ztrio)
                break
            except:
                pass

        return point

    def get_grad(self):
        func = sp.Matrix(self.function)
        grad = func.jacobian(self.affine_coordinates)
        return grad

    def get_hol_n_form(self, coord):
        """

        Return:
        -------
        A or a list of symbolic expressions of the holomorphic n-form 1/(∂f/∂z_i)

        """
        hol_n_form = []
        try:
            hol_n_form = 1/self.grad[:,coord].det()
        except:
            logging.exception('The number of functions and the number of coordinates to eliminate do not match')

        return hol_n_form

    def autopatch(self):
        # projective patches
        points_on_patch = [[] for i in range(self.n_dim)]
        for point in self.points:
            norms = np.absolute(point)
            for i in range(self.n_dim):
                if norms[i] == max(norms):
                    point_normalized = self.normalize_point(point, i)
                    points_on_patch[i].append(point_normalized)
                    continue
        for i in range(self.n_dim):
            self.set_patch(points_on_patch[i], i)

        # Remove empty patches
        self.patches = [subpatch for subpatch in self.patches if subpatch.points]

        for patch in self.patches:

            jac_det = []
            for i in range(self.n_dim-1):
              det_row = []
              for j in range(self.n_dim-1):
                det_row.append(patch.grad[:, [i,j]].det())
              jac_det.append(det_row)

            jac_det = sp.Matrix(jac_det)
            jac_det = sp.lambdify([self.coordinates], jac_det, 'numpy')

            jac_det_arr = np.abs(np.squeeze(np.vectorize(jac_det,signature='(n)->(p,q)')(patch.points))) 

            n, m, _ = jac_det_arr.shape
            # Reshape the array to a 2D array where each row represents one mxm subarray
            reshaped_arr = jac_det_arr.reshape(n, -1)
            # Find the argmax indices for each row (mxm subarray)
            argmax_indices = np.argmax(reshaped_arr, axis=1)
            # Convert the flat indices to row and column indices
            row_indices, col_indices = np.unravel_index(argmax_indices, (m, m))
            # Stack the row and column indices horizontally to get the final result
            result = np.column_stack((row_indices, col_indices))

            max_grad_list = np.unique(result, axis=0).tolist()

            points_arr = np.array(patch.points)
            for max_grad_coord in max_grad_list:
                points_on_patch = points_arr[np.where(np.all(result == max_grad_coord, axis=1))]
                patch.set_patch(points_on_patch, patch.norm_coordinate, max_grad_coord=max_grad_coord)

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
        # Since we have more than one ignored_coordinate in CICY, sympy subs()
        # cannot replace two coordinates simultaneously. As a result, if the first
        # expression contains the coordinate to be replaced by the second expression
        # that coordinate will also be replaced. So here we will create a temporary 
        # coordinate list W to avoid this issue
        W = sp.var('w0:{}'.format(len(self.affine_coordinates)))
        ignored_coordinate = np.array(W)[ignored_coord]
        local_coordinates = sp.Matrix(W).subs({coord: func for coord, func in zip(ignored_coordinate, self.function)})
        local_coordinates = local_coordinates.subs({w: z for w, z in zip(W, self.affine_coordinates)})
        restriction = local_coordinates.jacobian(self.affine_coordinates).inv()
        for coord in reversed(ignored_coord):
            restriction.col_del(coord)
        if lambdify is True:
            restriction = sp.lambdify([self.coordinates], restriction, 'numpy')
        return restriction

class RealCICYHypersurface(CICYHypersurface, RealHypersurface):

    def generate_random_projective(self, n_set, n_pt_in_a_set):
        return RealHypersurface.generate_random_projective(self, n_set, n_pt_in_a_set)

    def solve_points_multiprocessing(self, coeffs_list, monoms_list, ztrios):
        points = []
        with Pool() as pool:
            for point in pool.starmap(RealCICYHypersurface.solve_poly, zip(coeffs_list, monoms_list, ztrios)):
                points.append(point)

        points = list(filter(lambda x: x is not None, points))
        return points

    @staticmethod
    def solve_poly(coeff_list, monom_list, ztrio):
        point = None

        def func_t(t0, t1):
            return [sum([coeff * t0**monom[0]*t1**monom[1] for coeff, monom in zip(coeffs, monoms)]) for coeffs, monoms in zip(coeff_list, monom_list)]

        for attempt in range(1):
            try:
                #t_real = np.random.randn(4)
                #t_init = [complex(t_real[0], t_real[1]), complex(t_real[2], t_real[3])]
                t_init = np.random.randn(2).tolist()
                t_solved = mpmath.findroot(func_t, t_init)
                t_array = np.array(t_solved.tolist(), dtype=np.complex64)
                t_array = np.concatenate((t_array, np.array([[1.0+0.0j]])))
                point = np.add.reduce(t_array * ztrio)
                break
            except:
                pass

        return point
