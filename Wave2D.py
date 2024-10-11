import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation


x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.L = 1.0
        x = np.linspace(0,self.L, N+1)
        y = np.linspace(0,self.L, N+1)
        self.xij, self.yij = np.meshgrid(x,y, indexing="ij" )
    

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        self.kx = np.pi*self.mx
        self.ky = np.pi*self.my
        return self.c * np.sqrt(self.kx**2 + self.ky**2)
    
    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.Unp1, self.Un, self.Unm1 = np.zeros((3, N+1, N+1))
        self.Unm1[:] = sp.lambdify((x, y, t), self.ue(mx, my))(self.xij, self.yij, 0)
        self.Un[:] = self.Unm1[:] + 0.5*(self.c*self.dt)**2*(self.D @ self.Unm1 + self.Unm1 @ self.D.T)

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.dx/self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        u_j = sp.lambdify((x, y,t), self.ue(self.mx, self.my))(self.xij, self.yij, t0)
        return np.sqrt(self.h**2 * np.sum((u-u_j)**2))

    def apply_bcs(self):
        self.Unp1[0] = 0
        self.Unp1[-1] = 0
        self.Unp1[:, -1] = 0
        self.Unp1[:, 0] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N
        self.cfl = cfl
        self.c = c 
        self.mx = mx
        self.my = my
        self.dx = self.h = 1.0/N 
        self.D = self.D2(N)/self.dx**2

        self.create_mesh(N=N, sparse  =  False)
        self.initialize(N=N, mx = mx, my = my)

        self.plotdata = {}
        err = []
        err.append(self.l2_error(self.Un, self.dt))

        if store_data == 1:
            self.plotdata = self.Un.copy()

        for n in range(1, Nt):
            self.Unp1[:] = 2*self.Un - self.Unm1 + (self.c*self.dt)**2*(self.D @ self.Un + self.Un @ self.D.T)
            self.apply_bcs()
            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1
            if n % store_data == 0:
                self.plotdata[n] = self.Unm1.copy()
            
            err.append(self.l2_error(self.Un, self.dt*(n+1)))
            
        if store_data == -1:
            return self.h, err
        if store_data > 0:
            return self.plotdata




    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def test_convergence_wave2d():
        sol = Wave2D()
        r, E, h = sol.convergence_rates(mx=2, my=3)
        assert abs(r[-1]-2) < 1e-2

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        return D
    
    def ue(self, mx, my):
        return sp.cos(self.w*t)*sp.cos(mx*sp.pi * x)*sp.cos(my*sp.pi*y)

    def apply_bcs(self):
        pass

def test_convergence_wave2d():
    """
    Test the convergence rates of the Wave2D solver.

    This function initializes a Wave2D solver instance and computes the 
    convergence rates using specified grid resolutions. It then asserts 
    that the last computed convergence rate is close to 2 within a 
    tolerance of 1e-2.

    Raises:
        AssertionError: If the last computed convergence rate is not 
                        within the specified tolerance of 2.
    """
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    """
    Test the convergence rates of the Wave2D_Neumann solver.

    This function initializes the Wave2D_Neumann solver and computes the 
    convergence rates for a given mesh resolution. It then asserts that 
    the last computed convergence rate is approximately 2, within a 
    tolerance of 0.05.

    Raises:
        AssertionError: If the last convergence rate is not within the 
                        specified tolerance of 2.
    """
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    """
    Test the Wave2D and Wave2D_Neumann solvers for the 2D wave equation.
    This function initializes the solvers with a specific configuration and 
    checks if the error at the final time step is below a specified threshold.
    The test uses:
    - Grid size: 3x3 (mx = my = 3)
    - Courant number: C = 1/sqrt(2)
    - Time steps: 4000
    - Final time: 10
    The function asserts that the final error for both solvers is less than 1e-12.
    Raises:
        AssertionError: If the final error for either solver is not below 1e-12.
    """
    mx = my = 3
    C = 1/np.sqrt(2)

    sol = Wave2D()
    sol_neumann = Wave2D_Neumann()

    _, err1 = sol(4000, 10, C, mx, my, store_data = -1)
    _, err2 = sol_neumann(4000, 10, C, mx, my, store_data = -1)

    assert err1[-1] < 1e-12
    assert err2[-1] < 1e-12

def create_animation():
    """
    Generates a 2D wave animation and saves it as a GIF file.
    This function creates an instance of the Wave2D class, computes the wave 
    solution, and generates an animation of the wave propagation over time. 
    The animation is saved as a GIF file in the specified location.
    Parameters:
    None
    Returns:
    None
    Notes:
    - The function uses a 3D plot to visualize the wave surface.
    - The animation is created using matplotlib's ArtistAnimation.
    - The resulting GIF is saved in the 'report' directory with the filename 'wavemovie2d.gif'.
    """

    wave = Wave2D()
    solution = wave(N=100, Nt = 100, cfl = 1/np.sqrt(2), mx = 2, my = 2, store_data = 2)
    xij, yij = wave.xij, wave.yij
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    frames = []
    for n, val in solution.items():
        frame = ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2, cmap=plt.cm.viridis)
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True,
                                    repeat_delay=1000)
    ani.save('report/wavemovie.gif', writer='pillow', fps=5) 

if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    create_animation()
