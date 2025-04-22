"""kit for studying, debuging. But not needed for implimentation.

ref: [eth-6886-02]

functions  
==========
### utility functions, imported when `imoprt *`
    `generate_complex_2dcoord`: Generate mesh of complex coordinate, squared region.
    `generate_2d_gaussian`: Generates a 2D Gaussian distribution and its used coordinate.
    `plot_moments_bar_diagram`: plot the value of moments, up to 4-th moments.
    `plot_complex_2dfunc`: plot 2d function, with complex coordinate.
    `plot_complex_2dfunc_in3d`: Plot function as a 3D surface with a wireframe style.
    `eva_S_moment_intermsof_ah`: Returns binomal expension of ⟨S†^n S^m⟩.
    `eva_qubit_moments_intermsof_sh`: Evaluate qubit moments <a^†n a^m> up to a specific order, default is 4.
    `eva_fock_basis_expr`: Apply a and adag operator |n> and compute <n|m> to return final result.
    `eva_qubit_moment_by_ket`: Evaluate qubit moments <a^†n a^m> up to a specific order, default is 4.
"""

import sympy as sp
sp.init_printing()
from sympy.physics.quantum import Bra, Ket, Dagger, Operator
from sympy import Mul, sqrt

# anilation operator
a = Operator('a')
adag = Dagger(a)

import numpy as np
from matplotlib import pyplot as plt
__all__ = [
    'a',
    'adag',
    'generate_complex_2dcoord',
    'generate_2d_gaussian',
    'plot_moments_bar_diagram',
    'plot_complex_2dfunc',
    'plot_complex_2dfunc_in3d',
    'eva_S_moment_intermsof_ah',
    'eva_qubit_moments_intermsof_sh',
    'eva_fock_basis_expr',
    'eva_qubit_moment_by_ket'
]

def generate_complex_2dcoord(xy_range, n_pts):
    """Generate mesh of complex coordinate, squared region.
    
    Example usage:
    >>> generate_complex_2dcoord(1, 3)
    OUTPUT:
    |array([
    |   [-1.-1.j,  0.-1.j,  1.-1.j],
    |   [-1.+0.j,  0.+0.j,  1.+0.j],
    |   [-1.+1.j,  0.+1.j,  1.+1.j]
    |])
    """
    x = np.linspace(-xy_range, xy_range, n_pts)
    y = np.linspace(-xy_range, xy_range, n_pts)
    x_mesh, y_mesh = np.meshgrid(x, y)
    return x_mesh + 1j*y_mesh


def generate_2d_gaussian(means, sigmas, ranges, num_points=1024):
    """Generates a 2D Gaussian distribution and its used coordinate.
    (Geneate by AI)

    Args:
        means (length 2 list): The means [mean_x, mean_y] of gaussian.
        sigmas (length 2 list): standard deviations [sigma_x, sigma_y] of gaussian.
        ranges (length 2 list): The ranges [range_x, range_y] of gaussian.
        num_points (int): The number of points to generate along each dimension (default: 1024).

    Returns:
        gassain (2d array): Gaussian value correspond to `complex_2dcoord`.
        complex_2dcoord (2d array): The 2d grid of complex coordinates.
    """
    mean_x, mean_y = means
    sigma_x, sigma_y = sigmas
    range_x, range_y = ranges

    x = np.linspace(-range_x, range_x, num_points)
    y = np.linspace(-range_y, range_y, num_points)
    x, y = np.meshgrid(x, y)

    z = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
        -0.5 * (((x - mean_x) ** 2) / (sigma_x ** 2) + ((y - mean_y) ** 2) / (sigma_y ** 2))
    )
    return  z, x + 1j*y


def plot_moments_bar_diagram(moments:dict, title = 'title', func = np.abs):
    """plot the value of moments, up to 4-th moments.
    (Geneate by AI)
    
    Args:
        func (function): the function act on moment, default is np.abs
    """
    # add 0 as value if the key is not exist
    for n in range(5):
        for m in range(5):
            if n+m < 5:
                moments[f'a{n}{m}'] = moments.get(f'a{n}{m}', 0)

    data = func(np.array([
    [moments['a04'],             0,               0,              0,              0],
    [moments['a03'], moments['a13'],              0,              0,              0],
    [moments['a02'], moments['a12'], moments['a22'],              0,              0],
    [moments['a01'], moments['a11'], moments['a21'], moments['a31'],              0],
    [moments['a00'], moments['a10'], moments['a20'], moments['a30'], moments['a40']],
    ]))
    
    # Define grid size based on data dimensions
    n, m = data.shape

    # Create a mesh grid for the x and y coordinates
    x, y = np.meshgrid(np.arange(m), np.arange(n))

    # Flatten the mesh grid and data for plotting
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)  # Start bars at z = 0
    dx = dy = 0.5  # Width of bars
    dz = data.flatten()  # Heights of bars based on the data

    # Plotting
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D bar chart
    ax.bar3d(x, y, z, dx, dy, dz, 
             color=['white' if h == 0 else 'red' for h in dz], 
             alpha=0.5, 
             edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('n')
    ax.set_ylabel('m')

    # Set integer-only ticks on x and y axes
    ax.set_xticks(np.arange(0, n, 1))
    ax.set_yticks(np.arange(0, m, 1))
    ax.set_yticklabels(
        [''] + ['   ' + str(m - 1 - i) + '' for i in range(m-1)] , 
        rotation=60, 
        ha='right'
    )

    ax.grid(False)
    plt.show()


def plot_complex_2dfunc(func_value2d: np.ndarray, 
                        coord2d: np.ndarray,
                        title='title',
                        cmap='viridis'):
    """plot 2d function, with complex coordinate
    
    
    Args:
        func_value2d (2d numpy array): the function value to be ploy
        coord2d (2d numpy array): complex coordinate
    
    """
    x_mesh, y_mesh = np.real(coord2d), np.imag(coord2d)
    x_range = min(x_mesh[0, :]), max(x_mesh[0, :])
    y_range = min(y_mesh[:, 0]), max(y_mesh[:, 0])
    
    plt.imshow(func_value2d, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
               origin='lower', cmap=cmap, aspect='equal')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('P')
    plt.title(title)
    plt.show()


def plot_complex_2dfunc_in3d(func_value2d: np.ndarray, coord2d: np.ndarray, 
                             title='3D Plot', cmap='coolwarm',
                             elev=30, azim=45):
    """Plot a 3D surface with a wireframe style
    
    Args:
        func_value2d (2D numpy array): The function values to be plotted.
        coord2d (2D numpy array): Complex coordinate grid.
        elev, azim (float): view perspective angles, in degree. Default is 30, 45.
    """
    x_mesh, y_mesh = np.real(coord2d), np.imag(coord2d)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.view_init(elev=elev, azim=azim)
    # Plot the surface with wireframe
    ax.plot_surface(x_mesh, y_mesh, func_value2d, 
                    cmap=cmap, edgecolor='black', linewidth=0.6, alpha=0.7)

    # style
    ax.grid(True)
    ax.set_xlim([np.min(x_mesh), np.max(x_mesh)])
    ax.set_ylim([np.min(y_mesh), np.max(y_mesh)])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('P')
    ax.set_title(title)
    
    plt.show()


def eva_S_moment_intermsof_ah(n, m):
    """Returns binomal expension of ⟨S†^n S^m⟩, in terms of a and h.
    
    Explanation:
        ref:[eth-6886-02], p.55, eqa(3.23).
    
    Example usage:
    >>> eva_S_moment_intermsof_ah(1, 1)
    OUTPUT:
    |   ah + a†a + a†h† + hh†
    """
    
    a, adag, h, hdag = sp.symbols(r'a a^\dagger h h^\dagger', commutative=False)
    i, j = sp.symbols('i j')
    expr = sp.Sum(
        sp.binomial(m, j) * sp.binomial(n, i) \
        * adag**i * a**j * h**(n-i) * hdag**(m-j),
        (i, 0, n),
        (j, 0, m)
    )
    return expr.doit()


def eva_qubit_moments_intermsof_sh(subs_anm=False, highest_order=4) -> dict:
    """Evaluate qubit moments <a^†n a^m> up to a specific order, default is 4. in terms of S and h.
    
    Args:
        subs_anm (bool): Whether to expand <a^†n a^m> in terms of S, S†, h, h†
    
    Explanation:
        ref: [eth-6886-02], p.55, eqa(3.23). Solve moments from low order to \
        high order one after one.
    
    Example usage:
    >>> moments = eva_qubit_moments_intermsof_sh()
    >>> moments['a11']
    OUTPUT:
    |   S†S - (ah + a†h† + hh†)
    """
    S, Sdag, h, hdag = sp.symbols(r'S S^\dagger h h^\dagger', commutative=False)
    a, adag = sp.symbols(r'a a^\dagger', commutative=False)
    def eva_m_hnm_anti(n, m):
        return h**n * hdag**m
    def eva_m_Snm(n, m):
        return Sdag**n * S**m
    def eva_m_anm(n, m):
        to_be_subtract = 0
        for j in range(m+1):
            for i in range(n+1):
                if j==m and i==n:
                    pass
                else:
                    if not subs_anm:
                        of_a = adag**i * a**j 
                    else:
                        of_a = moments[f'a{i}{j}'] # moments['aij'] with i<n, j<m must exsit
                    of_h = eva_m_hnm_anti(n-i, m-j)
                    to_be_subtract += sp.binomial(n, i) * sp.binomial(m, j) * of_a * of_h
                    
        ans = eva_m_Snm(n, m) - to_be_subtract
        return ans
    
    moments = {}
    # moments['a00'] = sp.symbols('a_{00}', commutative=False) # special case
    # dev log: by trying above, and compare the result, found that moments['a00'] = 1 is correct.
    # originally I use moments['a00'] = eva_m_Snm(0, 0) and it is fiex by this trial.
    moments['a00'] = 1 # special case
    for order in range(1, highest_order + 1):
        for n in range(order, -1, -1):  # Iterate over (n, m) pairs
            m = order - n
            moments[f'a{n}{m}'] = eva_m_anm(n, m)
    return moments


def eva_fock_basis_expr(expr, dim:int = 4):
    """Apply a and adag operator |n> and compute <n|m> to return final result.

    Args:
        expr: the sympy expression that contains <n|, |n>, a, adag.
        dim (int): the Hilbert space dimension to be considered. i.e. use |0> ~ |dim-1>.

    >>> ket_state_expr = a * adag * a * (Ket(1) + Ket(0)) / sqrt(2)
    >>> ket_state = eva_fock_basis_expr(ket_state_expr, dim=2)
    >>> bra_state = (Bra(1) + Bra(0)) / sqrt(2)
    >>> inner_product = eva_fock_basis_expr(bra_state * ket_state, dim=2)
    >>> print(ket_state)
    >>> print(inner_product)
    OUTPUT:
    | |0> / sqrt(2)
    | 1 / 2
    """
    # list all  a|n>, adag|n> within our Hilber space
    a_adag_rules = {}
    for n in range(0, dim):
        a_adag_rules[a * Ket(n)] = sqrt(n) * Ket(n - 1)
        a_adag_rules[adag * Ket(n)] = sqrt(n + 1) * Ket(n + 1)
    a_adag_rules[a * Ket(0)] = 0        #    a|0>     = 0
    a_adag_rules[adag * Ket(dim-1)] = 0 # adag|n_max> = 0

    # substitute a|n>, adag|n>, repeatly untill it is non-changing
    a_applied_expr = expr.expand().subs(a_adag_rules)
    while expr != a_applied_expr:
        expr = a_applied_expr
        a_applied_expr = expr.expand().subs(a_adag_rules)

    # list all  <n|m> = delta_nm within our Hilber space
    nm_inner_prod_rules = {}
    for n in range(0, dim):
        for m in range(0, dim):
            # <n|m> and <n|*|m> expression are like sympy.physics.quantum bug
            if n == m:
                nm_inner_prod_rules[Bra(n) * Ket(m)] = 1 # <n|m>
                nm_inner_prod_rules[Mul(Bra(n), Ket(m))] = 1 # <n|*|m>
            else:
                nm_inner_prod_rules[Bra(n) * Ket(m)] = 0  # <n|m>
                nm_inner_prod_rules[Mul(Bra(n), Ket(m))] = 0 # <n|*|m>

    final_expr = a_applied_expr.expand().subs(nm_inner_prod_rules)
    return final_expr


def eva_qubit_moment_by_ket(ket_state, highest_order=4, dim=4):
    """Evaluate qubit moments <a^†n a^m> up to a specific order, default is 4.

    Args:
        ket_state: the sympy expression for ket state.
        highest_order (int): the highest_order of moment, default is 4.
        dim (int): the Hilbert space dimension to be considered. i.e. use |0> ~ |dim-1>.
    
    Example usage:
    >>> ket_state = (Ket(2) + Ket(0)) / sqrt(2)
    >>> eva_qubit_moment_by_ket(ket_state, highest_order=2)
    OUTPUT:
    | {'a00': 1, 'a01': 0, 'a02': sqrt(2)/2, 'a10': 0, 'a11': 1, 'a20': sqrt(2)/2}
    """
    bra_state = Dagger(ket_state)
    
    moments = {}
    for n in range(highest_order + 1):
        for m in range(highest_order + 1):
            if n + m < highest_order + 1:
                operator = adag**n * a**m
                moment = eva_fock_basis_expr(bra_state * operator * ket_state, dim=dim)
                moments[f'a{n}{m}'] = moment
    return moments
