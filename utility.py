"""Utilities that is helpful for implimentation of tomography

ref:
[eth-6886-02]:

"""

import numpy as np

def approx_complex_2dint(func2d: np.ndarray, coord2d: np.ndarray) -> float:
    """
    Approximates the 2D integral of a function using a discrete sum over a rectangular region.
    (generate by AI)
    
    Example usage:
    >>> # func to be integrate
    >>> def gaussian_2d(alpha):
    >>>     x, y = np.real(alpha), np.imag(alpha)
    >>>     return np.exp( - (x**2 / 2 + y**2 / 2) )
    >>> 
    >>> # create coord and obtain function values 
    >>> x_mesh, y_mesh = np.meshgrid(
    >>>     np.linspace(-5, 5, 201), # x
    >>>     np.linspace(-5, 5, 201), # y
    >>> )
    >>> coord2d = x_mesh + 1j*y_mesh
    >>> gaussian_values = gaussian_2d(coord2d)
    >>> 
    >>> # Compute approximate integral
    >>> integral_value = approx_complex_2dint(gaussian_values, coord2d)
    >>> print(integral_value)
    OUTPUT:
    | 6.283178998088227
    """
    x_mesh, y_mesh = np.real(coord2d), np.imag(coord2d)
    deltax = abs(x_mesh[0, 0] - x_mesh[0, 1])
    deltay = abs(y_mesh[0, 0] - y_mesh[1, 0])
    return np.sum(func2d) * deltax * deltay

def eva_S_moment(S: np.ndarray, n: int, m: int, D: np.ndarray):
    """Evaluate normally-ordered moment, ⟨S†^n S^m⟩, of histogram D.
    
    ref:[eth-6886-02], p.53, eqa(3.18).
    
    Args:
    -- S: 2d array. As coordinate.
    -- D: 2d array. The histograme, or denoted as D(S).
    
    It returns ⟨S†^n S^m⟩.
    """
    return approx_complex_2dint(S.conj()**n * S**m * D, S)

def eva_h_moment_anti(S: np.ndarray, n: int, m: int, D_h: np.ndarray):
    """Evaluate anti-ordered moment, ⟨h^n h†^m⟩, of histogram D_h.
    
    ref:[eth-6886-02], p.55, eqa(3.24).
    
    Args:
    -- S: 2d array. As coordinate.
    -- D_h: 2d array. The histograme of ref state, or denoted as D_h(S).
    """
    return eva_S_moment(S, n, m, D_h)

def eva_qubit_moments(S, D_S, D_h, highest_order=4) -> dict:
    """Evaluate qubit moments <a^†n a^m> up to a specific order, default is 4.
    
    Args:
    -- S: 2d array. As coordinate.
    -- D_S: 2d array. The histograme of measurment, or denoted as D_S(S).
    -- D_h: 2d array. The histograme of ref state, or denoted as D_h(S).
    
    Returns:
    -- A dictionary with key 'a01', 'a13', 'a22', etc...
    
    By using: [eth-6886-02], p.55, eqa(3.23). 
    Solve moments from low order to high order one after one.
    >>> moments = eva_qubit_moments(S, D_S, D_h, highest_order=4)
    >>> moments['a01']
    
    """
    from scipy.special import comb
    def eva_m_hnm_anti(n, m):
        return eva_h_moment_anti(S, n, m, D_h)
    def eva_m_Snm(n, m):
        return eva_S_moment(S, n, m, D_S)
    def eva_m_anm(n, m):
        to_be_subtract = 0
        for j in range(m+1):
            for i in range(n+1):
                if j==m and i==n:
                    pass
                else:
                    of_a = moments[f'a{i}{j}'] # moment['aij'] with i<n, j<m must exsit
                    of_h = eva_m_hnm_anti(n-i, m-j)
                    to_be_subtract += comb(n, i) * comb(m, j) * of_a * of_h

        ans = eva_m_Snm(n, m) - to_be_subtract
        return ans

    moments = {}
    moments['a00'] = 1 # special case
    for order in range(1, highest_order + 1):
        for n in range(order, -1, -1):  # Iterate over (n, m) pairs
            m = order - n
            moments[f'a{n}{m}'] = eva_m_anm(n, m)
    return moments




def get_winger_function_func(n_max, m_max, lambd, moments: dict):
    """Return a function that takes only one complex number input, alpha, to have W(alpha).
    
    By using: [eth-6886-02], p.59. 
    
    lambd: coord to be intergal
    
    Example usage:
    >>> winger_function = get_winger_function_func(2, 2, S, moments)
    >>> winger_function_value = winger_function(2 + 2j)
    """
    def winger_function(alpha):
        from math import factorial

        exp_term = np.exp(
            -1/2 * abs(lambd)**2 + alpha*np.conj(lambd) - np.conj(alpha)*lambd
        )

        frac_term = np.zeros_like(lambd, dtype=complex)
        for n in range(n_max):
            for m in range(m_max):
                moment = moments.get(f'a{n}{m}', 0) # for higher order, assume to be zero
                frac_term += moment * (-np.conj(lambd)**m * lambd**n) / (np.pi**2 * factorial(n) * factorial(m))
        return np.real(approx_complex_2dint(frac_term * exp_term, lambd) )
    
    return winger_function