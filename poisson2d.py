import numpy as np
import sympy as sp
from scipy import sparse
from scipy.interpolate import interpn

x, y = sp.symbols('x y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, N, ue, f=None):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : float
            The length of the domain in both x and y directions
        N : int
            Number of grid points in each direction
        ue : sympy expression
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        f : sympy expression, optional
            The source term in Poisson's equation. If None, it will be computed as the Laplacian of ue.
        """
        self.L = L
        self.N = N
        self.dx = L / N
        self.dy = L / N
        self.h = self.dx

        self.ue = ue

        x_sym, y_sym = sp.symbols('x y')

        if f is None:
            self.f = sp.diff(ue, x_sym, 2) + sp.diff(ue, y_sym, 2)
        else:
            self.f = f

        self.ue_func = sp.lambdify((x_sym, y_sym), ue, 'numpy')
        self.f_func = sp.lambdify((x_sym, y_sym), self.f, 'numpy')

        self.create_mesh(N)

    def create_mesh(self, N):
        """
        Create a 2D mesh grid and store it in self.x_grid and self.y_grid.

        Parameters
        ----------
        N : int
            Number of divisions in the grid for each direction.
        """
        x_vals = np.linspace(0, self.L, N + 1)
        y_vals = np.linspace(0, self.L, N + 1)
        self.x_grid, self.y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")

    def D2(self):
        """
        Generate a second-order differentiation matrix using central finite differences.

        Returns
        -------
        scipy.sparse.lil_matrix
            A sparse matrix representing the second-order differentiation.
        """
        size = self.N + 1
        diagonals = [np.ones(size - 1), -2 * np.ones(size), np.ones(size - 1)]
        offsets = [-1, 0, 1]
        D = sparse.diags(diagonals, offsets, shape=(size, size), format='lil')

        D[0, :4] = [2, -5, 4, -1]
        D[-1, -4:] = [-1, 4, -5, 2]

        return D

    def laplace(self):
        """
        Compute the vectorized Laplace operator using Kronecker products.

        Returns
        -------
        scipy.sparse.csr_matrix
            The Laplace operator in sparse matrix form.
        """
        D2_x = (1.0 / self.dx**2) * self.D2()
        D2_y = (1.0 / self.dy**2) * self.D2()

        I = sparse.eye(self.N + 1, format='csr')

        laplacian = sparse.kron(D2_x, I) + sparse.kron(I, D2_y)

        return laplacian.tocsr()

    def get_boundary_indices(self):
        """
        Get the indices of the vectorized grid points that are on the boundary.

        Returns
        -------
        numpy.ndarray
            Array of boundary indices.
        """
        boundary_mask = np.ones((self.N + 1, self.N + 1), dtype=bool)
        boundary_mask[1:-1, 1:-1] = False

        boundary_indices = np.flatnonzero(boundary_mask.ravel())

        return boundary_indices

    def assemble(self):
        """
        Assemble the system matrix A and the right-hand side vector b.

        Returns
        -------
        A : scipy.sparse.csr_matrix
            The system matrix after applying boundary conditions.
        b : numpy.ndarray
            The right-hand side vector.
        """
        A = self.laplace().tolil()

        ue_values = self.ue_func(self.x_grid, self.y_grid).ravel()
        f_values = self.f_func(self.x_grid, self.y_grid).ravel()

        b = f_values.copy()

        boundary_indices = self.get_boundary_indices()
        b[boundary_indices] = ue_values[boundary_indices]

        A[boundary_indices, :] = 0
        A[boundary_indices, boundary_indices] = 1

        A_csr = A.tocsr()

        return A_csr, b

    def l2_error(self, numerical_u):
        """
        Compute the L2 norm of the error between the numerical solution and the exact solution.

        Parameters
        ----------
        numerical_u : numpy.ndarray
            The numerical solution as a flattened array.

        Returns
        -------
        float
            The L2 norm of the error.
        """
        ue_values = self.ue_func(self.x_grid, self.y_grid)

        error_squared = (numerical_u.reshape((self.N + 1, self.N + 1)) - ue_values) ** 2

        l2_error = np.sqrt(np.sum(error_squared) * self.dx * self.dy)

        return l2_error

    def __call__(self, N=None):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int, optional
            The number of uniform intervals in each direction. If not provided, uses the initialized N.

        Returns
        -------
        numpy.ndarray
            The numerical solution as a 2D array.
        """
        if N is not None:
            self.N = N
            self.dx = self.dy = self.h = self.L / N
            self.create_mesh(N)
            
        A, b = self.assemble()

        self.U = sparse.linalg.spsolve(A, b).reshape((self.N + 1, self.N + 1))

        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int, optional
            The number of discretization levels to use (default is 6).

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing:
                - r: The convergence rates
                - E: The L2 errors
                - h: The mesh sizes
        """
        E = []
        h = []
        N_current = 8
        for _ in range(m):
            u = self(N_current)
            E.append(self.l2_error(u))
            h.append(self.h)
            N_current *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m)]
        return np.array(r), np.array(E[:-1]), np.array(h[:-1])

    def eval(self, x_val, y_val):
        """
        Evaluate the numerical solution u at arbitrary points (x, y) using bilinear interpolation.

        Parameters
        ----------
        x_val : float
            x-coordinate for evaluation.
        y_val : float
            y-coordinate for evaluation.

        Returns
        -------
        float
            The interpolated value of u at (x_val, y_val).

        Raises
        ------
        ValueError
            If the coordinates (x_val, y_val) are outside the domain [0, L].
        """
        if not (0 <= x_val <= self.L and 0 <= y_val <= self.L):
            raise ValueError("Coordinates (x, y) must be within the domain [0, L].")

        i = int(x_val // self.dx)
        j = int(y_val // self.dy)

        if i == self.N:
            i = self.N - 1
        if j == self.N:
            j = self.N - 1

        alpha = (x_val - i * self.dx) / self.dx
        beta = (y_val - j * self.dy) / self.dy

        u_ij = self.U[i, j]
        u_ip1j = self.U[i + 1, j]
        u_ijp1 = self.U[i, j + 1]
        u_ip1jp1 = self.U[i + 1, j + 1]

        interpolated_value = (
            (1 - alpha) * (1 - beta) * u_ij +
            alpha * (1 - beta) * u_ip1j +
            (1 - alpha) * beta * u_ijp1 +
            alpha * beta * u_ip1jp1
        )

        return interpolated_value

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    f = sp.diff(ue, x, 2) + sp.diff(ue, y, 2)
    sol = Poisson2D(L=1, N=100, ue=ue, f=f)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    f = sp.diff(ue, x, 2) + sp.diff(ue, y, 2)
    sol = Poisson2D(L=1, N=100, ue=ue, f=f)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h, y: 1-sol.h/2}).n()) < 1e-3

test_convergence_poisson2d()
test_interpolation()
