"""kits for studying, debuging, obtaining in-theory conclusion and plotting.

ref: [eth-6886-02]

This module provides tools for plotting the 2D functions and moments. 
And many kits to compute tomography obtained information theoretically,
can be used to evaluate fidelity, also helpful for debugging.

object and class
========
### sympy object
    `a`, `adag`: annihilation and creation operator, in sympy.

### sympy class
    `Bra`, `Ket`, `Dagger`: sympy quantum kit.

functions
==========
### plotting
    `plot_moments_bar_diagram`: plot the value of moments, up to 4-th moments.
    `plot_complex_2dfunc`: plot 2d function, with complex coordinate.
    `plot_complex_2dfunc_in3d`: Plot function as a 3D surface with a wireframe style.
    `plot_density_matrix_bar_diagram`: Plots a 3D bar diagram of a density matrix.
    `plot_moments_bar_diagram_flat`: Plot moment bar diagrme for real and imag part, as 2d bar diagrame.
    
### symbolic
    `eva_S_moment_intermsof_ah`: Returns binomal expension of ⟨S†^n S^m⟩.
    `eva_qubit_moments_intermsof_sh`: Evaluate qubit moments <a^†n a^m> up to a specific order, default is 4.
    `eva_fock_basis_expr`: Apply a and adag operator |n> and compute <n|m> to return final result.
    `eva_qubit_moments_by_ket`: Evaluate qubit moments <a^†n a^m> up to a specific order, default is 4.
    `eva_density_matrix_by_kets`:Evaluate density matrix by provide ket_states and probs to measure them. 
    
### 2D complex function utility
    `generate_complex_2dcoord`: Generate mesh of complex coordinate, squared region.
    `generate_2d_gaussian`: Generates a 2D Gaussian distribution and its used coordinate.

### misc
    `eva_qubit_moments_by_rho`: evaluate qubit moment by a given density matrix.
"""

import sympy as sp
sp.init_printing()
from sympy.physics.quantum import Bra, Ket, Dagger, Operator
from sympy import Mul, sqrt, Matrix

# annihilation operator
a = Operator('a')
adag = Dagger(a)

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

__all__ = [
    # sympy symbol and class
    'a',
    'adag',
    'Bra',
    'Ket',
    'Dagger',

    # plotting
    'plot_moments_bar_diagram',
    'plot_complex_2dfunc',
    'plot_complex_2dfunc_in3d',
    'plot_density_matrix_bar_diagram',
    'plot_moments_bar_diagram_flat',

    # symbolic
    'eva_S_moment_intermsof_ah',
    'eva_qubit_moments_intermsof_sh',
    'eva_fock_basis_expr',
    'eva_qubit_moments_by_ket',
    'eva_density_matrix_by_kets',
    
    # 2D complex function utility
    'generate_complex_2dcoord',
    'generate_2d_gaussian',
    
    # misc
    'generate_digitized_exp_decay_filter',
    'eva_qubit_moments_by_rho',
]


def generate_digitized_exp_decay_filter(decay_rate=0.15, num_points=30, y_shift=0, padding_front=10):
    """Create an exponential decay filter with tunable parameters.
    
    Args:
        num_points (int): Total Number of points of the filter.
        padding (int): Number of zero-padding points at the beginning.
    
    Returns:
        digitized_filter (ndarray): The generated exponential decay filter.
    """
    x = np.arange(num_points - padding_front)
    filter_values = np.exp(-decay_rate * np.maximum(0, x)) + y_shift
    padded_filter = np.concatenate((np.zeros(padding_front), filter_values))
    return padded_filter


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


def plot_moments_bar_diagram(
        moments: dict, 
        title='title', 
        phase_coloring: bool = True,
    ):
    """Plot the value of moments, up to 4-th moments with phase coloring and a colorbar.
    (Generate by AI)

    Args:
        moments (dict): dictionary of moments e.g. {'a00': value, 'a01': value, ...}
        title (str): title of the plot
        phase_coloring (bool): if true, plot in phase coloring. else in magnitude only
    """
    # ensure complex number, user might input sympy expression
    for key, value in moments.items():
        moments[key] = complex(value)

    # Fill missing moments with 0
    for n in range(5):
        for m in range(5):
            if n + m < 5:
                moments[f'a{n}{m}'] = moments.get(f'a{n}{m}', .0)

    # Build lower-triangular moment array
    moment_array = np.array([
        [moments['a04'],             0,               0,              0,              0],
        [moments['a03'], moments['a13'],              0,              0,              0],
        [moments['a02'], moments['a12'], moments['a22'],              0,              0],
        [moments['a01'], moments['a11'], moments['a21'], moments['a31'],              0],
        [moments['a00'], moments['a10'], moments['a20'], moments['a30'], moments['a40']],
    ])

    n, m = moment_array.shape
    x, y = np.meshgrid(np.arange(m), np.arange(n))
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)
    dx = dy = 0.5
    values = moment_array.flatten()
    dz = np.abs(values)
    phases = np.angle(values)

    # Normalize phase and create color map
    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    if phase_coloring:
        colors = plt.cm.hsv(norm(phases))
    else:
        colors = np.tile(np.array([1.0, 0.0, 0.0, 0.5]), (len(dz), 1))  # red with alpha 0.5


    # Compute mask for which bars should be hidden (n + m > 4)
    index_mask = np.zeros((5, 5), dtype=bool)
    for n in range(5):
        for m in range(5):
            if n + m > 4:
                index_mask[4 - n, m] = True
    index_mask = index_mask.flatten()

    # Adjust alpha and edgecolor for visibility
    alphas = np.where(index_mask, 0.0, 0.5)
    edgecolors = np.where(index_mask, '#F2F2F2', 'black')

    # Apply alpha to colors
    colors[:, -1] = alphas  # Modify alpha channel of RGBA colors

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(x)):
        ax.bar3d(x[i], y[i], z[i], dx, dy, dz[i],
                 color=colors[i],
                 edgecolor=edgecolors[i],
                 shade=True)

    ax.set_title(title)
    ax.set_xlabel('n')
    ax.set_ylabel('m')

    # Set integer-only ticks
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([''] + ['   ' + str(m - 1 - i) + '' for i in range(m - 1)],
                       rotation=60, ha='right')

    if phase_coloring:
        # Add colorbar for phase
        mappable = ScalarMappable(cmap='hsv', norm=norm)
        mappable.set_array([])
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label('Phase (radians)')
        cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    ax.grid(False)
    plt.show()


def plot_moments_bar_diagram_flat(
        moments: dict, 
        title='title', 
        highest_order: int = 4,
    ):
    """Plot moment bar diagrme for real and imag part, as 2d bar diagrame.
    (Generate by AI)

    Example usage:
    >>> plot_moments_bar_diagram_flat(moments, title='moments', highest_order=6)
    """
    # ensure complex number, user might input sympy expression
    for key, value in moments.items():
        moments[key] = complex(value)

    # Fill missing moments with 0
    for n in range(highest_order+1):
        for m in range(highest_order+1):
            if n + m < highest_order+1:
                moments[f'a{n}{m}'] = moments.get(f'a{n}{m}', .0)

    # Filter and sort keys by total order (n + m), then by n
    sorted_items = sorted(
        [(k, v) for k, v in moments.items() if int(k[1]) + int(k[2]) <= highest_order],
        key=lambda kv: (int(kv[0][1]) + int(kv[0][2]), int(kv[0][1]))
    )

    labels = [k[1:] for k, _ in sorted_items]
    values = np.array([v for _, v in sorted_items])
    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, np.real(values), width=width, label='Real', color='skyblue')
    plt.bar(x + width/2, np.imag(values), width=width, label='Imag', color='pink')

    # Add vertical dashed lines to separate total orders
    total_orders = [int(label[0]) + int(label[1]) for label in labels]
    prev_order = total_orders[0]
    for i in range(1, len(total_orders)):
        current_order = total_orders[i]
        if current_order != prev_order:
            plt.axvline(x=i - 0.5, color='gray', linestyle='--', linewidth=1)
            prev_order = current_order

    plt.xticks(x, labels)
    plt.title(title)
    plt.xlabel('$\\{n, m\\}$')
    plt.ylabel('$\\langle a^{\\dagger n} a^m\\rangle$')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(x, labels, rotation=90)
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
    
    fig = plt.figure()
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


def eva_qubit_moments_by_ket(ket_state, highest_order=4, dim=4):
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


def eva_density_matrix_by_kets(
        ket_states:list, 
        probs: list, 
        dim:int =4, 
        matrix_repr: bool=True
    ):
    """Evaluate density matrix by provide ket_states and probs to measure them. 

    Args:
        ket_states: an array of ket states.
        probs (int): an array of probability to find coorspond ket.
        dim (int): the Hilbert space dimension to be considered. i.e. use |0> ~ |dim-1>.
        matrix_repr (bool): return as matrix representation.
    
    Returns:
        density_matrix (sympy.matrix): the matrix representation of rho as sympy object, \
        one can use np.array(rho_sympy, dtype=complex) to convert it for other use.
        
    Explanation:
        For a system has p_i pf prbability to find the state in |i>, the density matrix \
        is given as rho = sum_i(p_i*|i><i|).

    Example usages:
    >>> ket_states = [
    >>>     (Ket(1) + Ket(2)) / sqrt(2),
    >>>     (Ket(0) - Ket(2)) / sqrt(2)
    >>> ]
    >>> probs = [0.3, 0.7]
    >>> rho_sympy = eva_density_matrix_by_kets(ket_states, probs, dim=3)
    >>> rho_numpy = np.array(rho_sympy, dtype=complex)
    >>> rho_sympy
    OUTPUT:
    | [ 0.35,    0  -0.35]
    | [    0, 0.15,  0.15]
    | [-0.35, 0.15,   0.5]
    """
    bra_states = [Dagger(ket_state) for ket_state in ket_states]
    density_matrix = 0
    for ket_state, bra_state, prob in zip(ket_states, bra_states, probs):
        # using Mul is for |n><m| is treated differently as |n>*<m| in sympy
        density_matrix += prob * Mul(ket_state, bra_state)
    
    if not matrix_repr:
        return density_matrix.expand()
    else:
        # compute matrix representation
        repr = {}
    for i in range(dim):
        vec = Matrix([[0] * dim])
        vec[0, i] = 1
        repr[Ket(i)] = vec.T
        repr[Bra(i)] = vec
    return density_matrix.subs(repr, simultaneous=True)


def eva_qubit_moments_by_rho(rho, highest_order=4):
    """evaluate qubit moment by a given density matrix."""
    from .postprocess import compute_tr_rho_adagn_am
    moments = {}
    for n in range(highest_order+1):
        for m in range(highest_order+1):
            if n+m < highest_order+1:
                moments[f'a{n}{m}'] = compute_tr_rho_adagn_am(rho, n, m) 
    return moments


def plot_density_matrix_bar_diagram(rho, title: str='title', phase_coloring: bool = True):
    """Plots a 3D bar diagram of a density matrix.
    (Generate by AI)
    """
    # ensure numpy array, user might input sympy array
    rho = np.array(rho, dtype=complex)

    rho = np.flip(rho, axis=0)
    n = rho.shape[0]
    assert rho.shape == (n, n), "Input must be a square matrix"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xpos, ypos = np.meshgrid(np.arange(n), np.arange(n))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    dz = np.abs(rho.flatten())
    phases = np.angle(rho.flatten())
    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    if phase_coloring:
        colors = plt.cm.hsv(norm(phases))
    else:
        colors = np.tile(np.array([1.0, 0.0, 0.0, 0.5]), (len(dz), 1))  # red with alpha 0.5

    ax.bar3d(xpos, ypos, zpos, dx=0.8, dy=0.8, dz=dz, 
             color=colors, edgecolor='black', shade=True, alpha=0.5)

    ax.set_xlabel('$|n\\rangle$')
    ax.set_ylabel('$\\langle m|$')
    ax.set_title(title)

    # Set integer ticks for x and y axes
    ax.set_xticks(np.arange(0, n, 1))
    ax.set_yticks(np.arange(0, n, 1))
    ax.set_yticklabels(
        [''] + ['   ' + str(n - 1 - i) + '' for i in range(n-1)] , 
        rotation=60, 
        ha='right'
    )

    if phase_coloring:
        # Add colorbar for phase
        mappable = ScalarMappable(cmap='hsv', norm=norm)
        mappable.set_array([])
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label('Phase (radians)')
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    plt.show()