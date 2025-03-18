"""kit for studying, debuging. But not needed for implimentation.

"""


import sympy as sp
#sp.init_printing()

import numpy as np
from matplotlib import pyplot as plt


def plot_moment_bar_diagram(moment, title = 'title'):
    """plot the absolute value of moments, up to 4-th moments.
    
    
    """
    
    data = np.abs(np.array([
    [moment['a04'],             0,             0,             0,             0],
    [moment['a03'], moment['a13'],             0,             0,             0],
    [moment['a02'], moment['a12'], moment['a22'],             0,             0],
    [moment['a01'], moment['a11'], moment['a21'], moment['a31'],             0],
    [moment['a00'], moment['a10'], moment['a20'], moment['a30'], moment['a40']],
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
    ax.set_zlabel('Value')

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
    

def generate_2d_gaussian(means, sigmas, ranges, num_points=1024):
    """Generates a 2D Gaussian distribution.

    Args:
        means: A list or numpy array of length 2 representing the means [mean_x, mean_y].
        sigmas: A list or numpy array of length 2 representing the standard deviations [sigma_x, sigma_y].
        ranges: A list or numpy array of length 2 representing the ranges [range_x, range_y].
        num_points: The number of points to generate along each dimension (default: 1024).

    Returns:
        D, S: numpy arrays of Gaussian value and complex coordinates.
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

def plot_complex_2dfunc(func2d: np.ndarray, 
                        coord2d: np.ndarray,
                        title='title'):
    """
    
    
    args:
    -- func: the function
    -- coord2d:  complex coordinate
    
    
    """
    x_mesh, y_mesh = np.real(coord2d), np.imag(coord2d)
    x_range = min(x_mesh[0, :]), max(x_mesh[0, :])
    y_range = min(y_mesh[:, 0]), max(y_mesh[:, 0])
    
    plt.imshow(func2d, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
               origin='lower', cmap='bwr', aspect='equal')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('P')
    plt.title(title)
    plt.show()

def sym_eva_S_moment(n, m):
    """Returns binomal expension of ⟨S†^n S^m⟩.
    
    ref:[eth-6886-02], p.55, eqa(3.23).
    
    Example usage:
    >>> sym_eva_S_moment(1, 1)
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

def sym_eva_qubit_moments(subs_anm=False, highest_order=4) -> dict:
    """Evaluate qubit moments <a^†n a^m> up to a specific order, default is 4.
    Returns a dictionary with key 'a01', 'a13', 'a22', etc...
    
    
    For `subs_anm=True`, the expression of <a^†n a^m> will be expended in terms of S, S†, h, h†
    
    By using: [eth-6886-02], p.55, eqa(3.23). 
    Solve moments from low order to high order one after one.
    
    Example usage:
    >>> moments = sym_eva_qubit_moments()
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