"""Utilities that is helpful for implimentation of tomography.

ref: [eth-6886-02]


functions
==========
### Temporal mode matching
    `temporal_mode_matching`: Try align the single and filter in time domain, return the best mathcing result.

###  moments
    `approx_complex_2dint`: Approximates the 2D integral of a function using a discrete sum over a rectangular region.
    `eva_qubit_moments`: Evaluate qubit moments <a^†n a^m> up to a specific order, default is 4.
 
### Winger function
    `get_winger_function_func`: Return a function that can take complex number(s) as input to return winger function value.
 
### density matrix
    `mle_density_matrix`: Find best suitable density matrix to a set of moments using Maximum Likelihood Estimation (MLE).
    `compute_fidelity`: Compute the fidelity between two density matrices.

### helper functions (not imporeted when `import *`)
    `eva_S_moment`: Evaluate normally-ordered moment, ⟨S†^n S^m⟩, of histogram D.
    `eva_h_moment_anti`: Evaluate anti-ordered moment, ⟨h^n h†^m⟩, of histogram D_h.
    `get_annihilation_operator`: Return matirx representation of annihilation operator a in `dim` dimensional Hilbert space.
    `compute_tr_rho_adagn_am`: Computes Tr[rho a†^n a^m] for a given density matrix `rho`.
    `project_to_density_matrix`: A projection to ensure a Hermitian matrix is a valid density matrix.
"""

# The docsting habit, it is dynamic, so not every parameter need to 
# be listed, nor all every function need all the headers.
def docstring_example():
    """General description about the function.
    (supplementary information)
    
    Args:
        parm1 (dtype): description of parm1.
        parm2 (dtype): description of parm2.

    Returns:
        return1 (dtype): description of return1.
        return2 (dtype): description of return2.

    Explanation:
        An short articel about the function, \
        may across mutiple lines.

    Example usage:
    >>> 
    >>> 
    >>>
    OUTPUT:
    |
    |
    |
        
    """

# helper functions are not imported when `import *`
__all__ = [
    ## Temporal mode matching
    'temporal_mode_matching',

    ## Evaluate moments
    'approx_complex_2dint',
    'eva_qubit_moments',

    ## Winger function
    'get_winger_function_func',

    ## Density matrix
    'mle_density_matrix',
    'compute_fidelity',
]

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import sqrtm


def temporal_mode_matching(digitized_single: np.ndarray, 
                           digitized_filter: np.ndarray) -> complex:
    """Try align the single and filter in time domain, return the best mathcing result.
    
    Returns:
        filtered_value (complex number):

    Explanation:
        The `digitized_single` and `digitized_filter` are convolved without reversed, for convolved value to be 
        largest, it is taken to be the single and filter matched in the time domain (temporal mode matching).
    """
    correlation_result = np.correlate(digitized_single, digitized_filter, mode='valid')
    return np.max(correlation_result)


def approx_complex_2dint(func_value2d: np.ndarray, coord2d: np.ndarray) -> complex:
    """Approximates the 2D integral of a function using a discrete sum over a rectangular region.
    (generate by AI)
    
    Returns:
        approxed_int_value (complex number):

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
    >>> coord2d = x_mesh + 1j*y_mesh # or use `generate_complex_2dcoord`
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
    return np.sum(func_value2d) * deltax * deltay


def eva_S_moment(S: np.ndarray, n: int, m: int, D: np.ndarray, G: float=1) -> complex:
    """Evaluate normally-ordered moment, ⟨S†^n S^m⟩, of histogram D.
    
    Args:
        S (2d numpy array): As coordinate.
        D (2d numpy array): The histograme, or denoted as D(S).
        G (float): Gain of the amplifier chain, default is 1.
        
    Returns:
        S_moment (complex number): The ⟨S†^n S^m⟩ value.

    Explanation:
        ref:[eth-6886-02], p.53, eqa(3.18).
    """
    gain_term = G ** (-(n+m)/2)
    func = S.conj()**n * S**m * D * gain_term
    return approx_complex_2dint(func, S)


def eva_h_moment_anti(S: np.ndarray, n: int, m: int, D_h: np.ndarray, G: float=1) -> complex:
    """Evaluate anti-ordered moment, ⟨h^n h†^m⟩, of histogram D_h.

    Args:
        S (2d numpy array): As coordinate.
        D (2d numpy array): The histograme, or denoted as D(S).
        G (float): Gain of the amplifier chain, default is 1.

    Returns:
        h_moment_anti (complex number): The ⟨h^n h†^m⟩ value.

    Explanation:
        ref:[eth-6886-02], p.55, eqa(3.24).
    """
    return eva_S_moment(S, n, m, D_h, G)


def eva_qubit_moments(S: np.ndarray, D_S: np.ndarray, D_h: np.ndarray, 
                      G: float=1, highest_order: int=4) -> dict:
    """Evaluate qubit moments <a^†n a^m> up to a specific order, default is 4.
    
    Args:
        S (2d numpy array): As coordinate.
        D_S (2d numpy array): The histograme of measurment, or denoted as D_S(S).
        D_h (2d numpy array): The histograme of ref state, or denoted as D_h(S).
        G (float): Gain of the amplifier chain, default is 1.
        highest_order (int): highest order, i.e. (n+m) value.

    Returns:
        moments (dict): A dictionary with key 'a01', 'a13', 'a22', etc...

    Explanation:
        By using: [eth-6886-02], p.55, eqa(3.23). 
        Solve moments from low order to high order one after one.
    
    Example usage:
    >>> moments = eva_qubit_moments(S, D_S, D_h, G=1, highest_order=4)
    >>> moments['a01']
    """
    # <h^n a^†m> will be used many times, like <a^†n a^m> does. So write
    # a function to reduce duplicate intergal computing.
    h_moments_anti = {}
    def get_h_moment_anti(n, m):
        """Return <h^n a^†m> value, evaulate it if not computed before."""
        if f'h{n}{m}' not in h_moments_anti:
            h_moments_anti[f'h{n}{m}'] = eva_h_moment_anti(S, n, m, D_h, G)
        return h_moments_anti[f'h{n}{m}']
    
    # Compute <a^†n a^m>, by subtacting <S^†n S^m> by all other terms expect <a^†n a^m>
    # in the expansion of [eth-6886-02], p.55, eqa(3.23).
    from scipy.special import comb
    def eva_m_anm(n, m):
        to_be_subtract = 0
        for j in range(m+1):
            for i in range(n+1):
                if j==m and i==n:
                    pass
                else:
                    of_a = qubit_moments[f'a{i}{j}'] # moment['aij'] with i<n, j<m must exsit
                    of_h = get_h_moment_anti(n-i, m-j)
                    to_be_subtract += comb(n, i) * comb(m, j) * of_a * of_h

        ans = eva_S_moment(S, n, m, D_S, G) - to_be_subtract
        return ans

    # compute <a^†n a^m> from low order to high order
    qubit_moments = {}
    qubit_moments['a00'] = 1 # special case
    for order in range(1, highest_order + 1):
        for n in range(order, -1, -1):  # Iterate over (n, m) pairs
            m = order - n
            qubit_moments[f'a{n}{m}'] = eva_m_anm(n, m)
    return qubit_moments


def get_winger_function_func(moments: dict, lambd: np.ndarray, highest_order: int=4):
    """Return a function that can take complex number(s) as input to return winger function value.

    Args:
        moments (dict): moments, a dict with keys 'a01', 'a11', etc...
        lambd (2d numpy array): coord to be integrated.
        highest_order (int): maxima moment order take into account
    
    Returns:
        winger_function (function): A function take can take complex number(s) as input.
        and return the winger function value.
        
    Explanation:
        ref: [eth-6886-02], p.59. 
    
    Example usage:
    >>> lambd = generate_complex_2dcoord(5, 50) # generally good enough
    >>> W = get_winger_function_func(moments, lambd, highest_order=4)
    >>> # single complex number as parm
    >>> value = W(2 + 2j)
    >>> # complex number mesh as param
    >>> alpha = generate_complex_2dcoord(2, 150)
    >>> value2d = W(alpha)
    """
    # ensure complex number, user might input sympy expression
    for key, value in moments.items():
        moments[key] = complex(value)

    # fraction term is indepedent of alpha, precompute it
    from math import factorial
    frac_term = np.zeros_like(lambd, dtype=complex)
    for n in range(highest_order+1):
        for m in range(highest_order+1):
            moment = moments.get(f'a{n}{m}', 0) # for higher order or missing ones, assume to be zero
            frac_term += moment * ( (-np.conj(lambd))**m * lambd**n) / (np.pi**2 * factorial(n) * factorial(m))
    
    # precompute delta A, for approximate intergal
    x_mesh, y_mesh = np.real(lambd), np.imag(lambd)
    deltax = abs(x_mesh[0, 0] - x_mesh[0, 1])
    deltay = abs(y_mesh[0, 0] - y_mesh[1, 0])
    delta_A = deltax * deltay
    
    def winger_function(alpha: np.ndarray | complex) -> np.ndarray | complex:
        """Returns W(alpha) based on moments provide to `get_winger_function_func`.
        
        Example usage:
        >>> # single complex number as parm
        >>> value = W(2 + 2j)
        >>> # complex number mesh as param
        >>> alpha = generate_complex_2dcoord(2, 150)
        >>> value2d = W(alpha)
        """
        if isinstance(alpha, np.ndarray):
            # for complex number mesh as param: flatten, brocast, then reshape
            original_shape = alpha.shape
            alpha_flattened = alpha.reshape(-1)
            alpha_reshaped = alpha_flattened[:, np.newaxis, np.newaxis]
            exp_term = np.exp(
                -1/2 * abs(lambd)**2 + alpha_reshaped*np.conj(lambd) - np.conj(alpha_reshaped)*lambd
            )
            winger_function_values = np.real(np.sum(frac_term * exp_term, axis=(1, 2)) * delta_A)
            return winger_function_values.reshape(original_shape)
        else:
            exp_term = np.exp(
            -1/2 * abs(lambd)**2 + alpha*np.conj(lambd) - np.conj(alpha)*lambd
            )
            return np.real(np.sum(frac_term * exp_term) * delta_A)
    
    return winger_function


def get_annihilation_operator(dim: int) -> np.ndarray:
    """Return matirx representation of annihilation operator a in `dim` dimensional Hilbert space.
    (Generate by AI)

    Returns:
        annihilation_operator(2d numpy array): The matrix representation of annihilation operator.

    Example usage:
    >>> get_annihilation_operator(4)
    OUTPUT:
    | array([
    | [0,      1,      0,      0],
    | [0,      0, 1.4142,      0],
    | [0,      0,      0, 1.7321],
    | [0,      0,      0,      0]
    | ])
    """
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n-1, n] = np.sqrt(n)
    return a


def compute_tr_rho_adagn_am(rho: np.ndarray, n: int, m: int) -> complex:
    """Computes Tr[rho a†^n a^m] for a given density matrix `rho`.
    (Geneate by AI)

    Returns:
        result (complex number): The Tr[rho a†^n a^m] value.

    Explanation:
        In quantum physics, Tr[rho A] is the expectation value of A on the \
        state represented by density matirx rho.

    Example usage:
    >>> rho = np.array([
    >>>     [0.5, 0, 0, 0, 0],
    >>>     [0, 0.3, 0, 0, 0],
    >>>     [0, 0, 0.2, 0, 0],
    >>>     [0, 0,  0,  0, 0],
    >>>     [0, 0,  0,  0, 0]
    >>> ])
    >>> compute_tr_rho_adagn_am(rho, 1, 1)
    OUTPUT:
    | 0.7 + 0j
    """
    
    # get matirx representation of a and adag
    dim = rho.shape[0]
    a = get_annihilation_operator(dim)
    adag = a.conj().T 

    # Compute the expectation value: Tr[rho a†^n a^m]
    # dev log: CANNOT USE `adag ** n`.
    adag_n = np.linalg.matrix_power(adag, n)
    a_m = np.linalg.matrix_power(a, m)
    expectation_value = np.trace(rho @ adag_n @ a_m)
    #print(expectation_value)
    return expectation_value


def project_to_density_matrix(matrix: np.ndarray) -> np.ndarray:
    """A projection to ensure a Hermitian matrix is a valid density matrix.
    (Generate by AI)

    Returns:
        density_matirx (2d numpy array): the density matirx that is ensured to be \
        hermitien and trace to be 1.

    Explanation:
        This method is introduced by chatGPT, but I didn't dive into the detail.\
        One can copy this and ask chatGPT again :). So far I know it:
        1. force it to be hermitian
        2. force its trace to be 1
        3. have used a technic called `Cholesky decomposition`.
    """
    matrix = (matrix + matrix.conj().T) / 2  # Ensure Hermitian
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 0, None)  # Ensure non-negative eigenvalues
    H_proj = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return H_proj / np.trace(H_proj)  # Normalize to ensure Tr(rho) = 1


def negative_log_likelihood(rho_flatten, dim, n_max, m_max, moments, stddevi):
    """Negative log-likelihood function for optimization.
    
    Explanation:
        ref: [eth-6886-02], page 60, eqa (3.30).
        I take negative sign to the function, then the "maximaize" task is then \
        becomes "minimize", which has much more algorithm to do it.
    """
    # devlog: scipy can't use complex, so replace code to do it by real and imag seperately
    ## rho_matrix = rho_flatten.reshape(dim, dim)  # Convert vector back to matrix
    rho_matrix = rho_flatten[:dim**2].reshape((dim, dim)) + 1j * rho_flatten[dim**2:].reshape((dim, dim))
    rho_matrix = project_to_density_matrix(rho_matrix)  # Ensure valid density matrix

    func_value = 0
    for n in range(n_max+1):
        for m in range(m_max+1):
            delta_value = stddevi.get(f'd{n}{m}', 1)
            moment_value = moments.get(f'a{n}{m}', 0)
            trace_term = compute_tr_rho_adagn_am(rho_matrix, n, m)
            # take negative sing so is use +=
            func_value += (1 / delta_value**2) * (np.abs(moment_value - trace_term) ** 2)
    return func_value


def mle_density_matrix(moments: dict, dim: int, 
                       highest_order: int =4, 
                       stddevi: dict={}) -> np.ndarray:
    """Find best suitable density matrix to a set of moments using Maximum Likelihood Estimation (MLE).
    (Generate by AI)
    
    Args:
        moment (dict): a dict with keys 'a01', 'a02', etc..
        dim (int): dimension of density matrix.
        highest_order (int): maxima moment order take into account
        stddevi (dict): stander deviation of moment measurment, all be 1 for default.

    Returns:
        density_matirx (2d numpy array): The MLE result of density matrix.

    Explanation:
        ref: [eth-6886-02], page 60, eqa (3.30).
        I take negative sign to the likelihood function, then the "maximaize" task is then \
        become "minimize", which has much more algorithm to do it.

    Exmaple usage:
    >>> moments = {'a00': 1, 'a11': 1} # single photon state |1>
    >>> mle_density_matrix(moments, dim=2)
    OUTPUT:
    | array([
    |     [ 3.08148793e-17, -5.55111514e-09],
    |     [-5.55111514e-09,  1.00000000e+00]
    | ])
    """
    # ensure complex number, usser might input sympy expression
    for key, value in moments.items():
        moments[key] = complex(value)

    # devlog: scipy can't use complex, so replace code to do it by real and imag seperately
    initial_rho = np.ones([dim, dim], dtype=complex) / dim  # Start with max coherence state
    ## initial_rho_vector = initial_rho.flatten()  # Flatten for optimization
    initial_rho_vector = np.concatenate([initial_rho.real.flatten(), initial_rho.imag.flatten()])

    n_max = m_max = highest_order   
    result = minimize(
        negative_log_likelihood, initial_rho_vector,
        args=(dim, n_max, m_max, moments, stddevi),
        method='L-BFGS-B'
    )

    # devlog: scipy can't use complex, so replace code to do it by real and imag seperately
    ## optimized_rho = result.x.reshape(dim, dim)  # Reshape back to matrix form 
    optimized_rho = result.x[:dim**2].reshape((dim, dim)) + 1j * result.x[dim**2:].reshape((dim, dim))
    optimized_rho = project_to_density_matrix(optimized_rho)
    return optimized_rho


def compute_fidelity(rho1: np.ndarray, rho2: np.ndarray) -> dict:
    """Compute fidelity between two density matrices.
    (Generate by AI)

    Example usage:
    >>> rho1 = np.array([[0.7, 0.3], [0.3, 0.3]])
    >>> rho2 = np.array([[0.6, 0.4], [0.4, 0.4]])
    >>> compute_fidelity(rho1, rho2)
    OUTPUT:
    | 0.975959179422654
    """
    # ensure numpy array
    rho1 = np.array(rho1, dtype=complex)
    rho2 = np.array(rho2, dtype=complex)

    # Fidelity: F(rho1, rho2) = (Tr(sqrt(sqrt(rho1) * rho2 * sqrt(rho1))))^2
    sqrt_rho1 = sqrtm(rho1)
    inner = np.trace(sqrtm(sqrt_rho1 @ rho2 @ sqrt_rho1))
    fidelity = np.abs(inner) ** 2  # Ensuring numerical stability

    return fidelity
