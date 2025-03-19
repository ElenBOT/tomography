"""Utilities that is helpful for implimentation of tomography

ref:
[eth-6886-02]:

"""

__all__ = [
    'approx_complex_2dint',
    'eva_S_moment',
    'eva_h_moment_anti',
    'eva_qubit_moments',
    'get_winger_function_func',
    'compute_tr_rho_adagn_am',
    'project_to_density_matrix',
    'negative_log_likelihood',
    'fit_density_matrix',
    'compute_validity',
]

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import sqrtm


def approx_complex_2dint(func2d: np.ndarray, coord2d: np.ndarray) -> float:
    """Approximates the 2D integral of a function using a discrete sum over a rectangular region.
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
    
    args:
    -- lambd: coord to be integrated.
    -- n_max, m_max: maxima n, m take into account
    
    By using: [eth-6886-02], p.59. 
    
    Example usage:
    >>> lambd = generate_complex_2dcoord(5, 50) # generally good enough
    >>> W = get_winger_function_func(2, 2, lambd, moments)
    >>> # single complex number as parm
    >>> value = W(2 + 2j)
    >>> # complex number mesh as param
    >>> alpha = generate_complex_2dcoord(2, 150)
    >>> value2d = W(alpha)
    """
    
    # fraction term is indepedent of alpha, precompute it
    from math import factorial
    frac_term = np.zeros_like(lambd, dtype=complex)
    for n in range(n_max):
        for m in range(m_max):
            moment = moments.get(f'a{n}{m}', 0) # for higher order, assume to be zero
            frac_term += moment * (-np.conj(lambd)**m * lambd**n) / (np.pi**2 * factorial(n) * factorial(m))
    
    # precompute delta A, for approximate intergal
    x_mesh, y_mesh = np.real(lambd), np.imag(lambd)
    deltax = abs(x_mesh[0, 0] - x_mesh[0, 1])
    deltay = abs(y_mesh[0, 0] - y_mesh[1, 0])
    delta_A = deltax * deltay
    
    def winger_function(alpha: np.ndarray | complex):
        """Returns W(alpha) based on moment provide to `get_winger_function_func`.
        
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
            winger_function_values = np.sum(frac_term * exp_term, axis=(1, 2)) * delta_A
            return winger_function_values.reshape(original_shape)
        else:
            exp_term = np.exp(
            -1/2 * abs(lambd)**2 + alpha*np.conj(lambd) - np.conj(alpha)*lambd
            )
            return np.sum(frac_term * exp_term) * delta_A
    
    return winger_function


def get_annihilation_operator(dim):
    """Return matirx representation of annihilation operator a in `dim` dimensional Hilbert space.
    (Generate by AI)

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


def compute_tr_rho_adagn_am(rho: np.ndarray, n: int, m: int):
    """Computes Tr[rho a†^n a^m] for a given density matrix `rho`.
    (Geneate by AI)

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

    In quantum physics, Tr[rho A] is the expectation value of A on the 
    state represented by density matirx rho.
    """
    
    # get matirx representation of a and adag
    dim = rho.shape[0]
    a = get_annihilation_operator(dim)
    adag = a.conj().T  

    # Compute the expectation value: Tr[rho a†^n a^m]
    operator = np.dot(adag ** n, a ** m)
    expectation_value = np.trace(np.dot(rho, operator))
    
    return expectation_value


def project_to_density_matrix(H):
    """A projection to ensure a Hermitian matrix is a valid density matrix.
    (Generate by AI)
    
    This method is introduced by chatGPT, but I didn't dive into the detail.
    One can copy this and ask chatGPT again :). So far I know it:
    1. force it to be hermitian
    2. force its trace to be 1
    3. have used a technic called `Cholesky decomposition`.
    """
    H = (H + H.conj().T) / 2  # Ensure Hermitian
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.clip(eigvals, 0, None)  # Ensure non-negative eigenvalues
    H_proj = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return H_proj / np.trace(H_proj)  # Normalize to ensure Tr(rho) = 1


def negative_log_likelihood(rho_flatten, dim, n_max, m_max, moments, stddevi):
    """Negative log-likelihood function for optimization."""
    rho_matrix = rho_flatten.reshape(dim, dim)  # Convert vector back to matrix
    rho_matrix = project_to_density_matrix(rho_matrix)  # Ensure valid rho
    
    func_value = 0
    for n in range(n_max):
        for m in range(m_max):
            delta_value = stddevi.get(f'd{n}{m}', 1)
            moment_value = moments.get(f'a{n}{m}', 0)
            trace_term = compute_tr_rho_adagn_am(rho_matrix, n, m)
            func_value -= (1 / delta_value**2) * np.abs(moment_value - trace_term) ** 2
    return func_value

def fit_density_matrix(dim, n_max, m_max, moments, stddevi):
    """Optimize the density matrix using Maximum Likelihood Estimation (MLE).
    (Generate by AI)
    
    Exmaple usage:
    >>> dim = 3  # Dimension of density matrix
    >>> n_max, m_max = 3, 3 # maximan moment order take into account
    >>> moments = {'a00': 1, 'a10': 0.5, 'a01': 0.4, 'a11': 0.2}  # Example moment data
    >>> stddevi = {'d00': 1, 'd10': 1, 'd01': 1, 'd11': 1} # Assume std deviation = 1 for now
    >>> fit_density_matrix(dim, n_max, m_max, moments, stddevi)
    OUTPUT:
    | array([
    |     [0.18672868, 0.23730362, 0.30910851],
    |     [0.23730362, 0.30157665, 0.39282969],
    |     [0.30910851, 0.39282969, 0.51169468]
    | ])
    """
    initial_rho = np.eye(dim) / dim  # Start with a maximally mixed state
    initial_rho_vector = initial_rho.flatten()  # Flatten for optimization
    
    result = minimize(
        negative_log_likelihood, initial_rho_vector, 
        args=(dim, n_max, m_max, moments, stddevi),
        method='L-BFGS-B'
    )
    
    optimized_rho = result.x.reshape(dim, dim)  # Reshape back to matrix form
    return project_to_density_matrix(optimized_rho)  # Ensure valid density matrix

def compute_similarities(rho1: np.ndarray, rho2: np.ndarray) -> dict:
    """Compute the similarities (Fidelity, trace Distance, Hilbert-Schmidt Distance) between two density matrices.
    (Generate by AI)

    Example usage:
    >>> rho1 = np.array([[0.7, 0.3], [0.3, 0.3]])
    >>> rho2 = np.array([[0.6, 0.4], [0.4, 0.4]])
    >>> compute_similarities(rho1, rho2)
    OUTPUT:
    | {'Fidelity': 0.975959179422654,
    |  'Trace Distance': 0.14142135623730953, 
    |  'Hilbert-Schmidt Distance': 0.04}
    """
    # Ensure Hermitian
    rho1 = (rho1 + rho1.conj().T) / 2
    rho2 = (rho2 + rho2.conj().T) / 2

    # Fidelity: F(rho1, rho2) = (Tr(sqrt(sqrt(rho1) * rho2 * sqrt(rho1))))^2
    sqrt_rho1 = sqrtm(rho1)
    fidelity = np.trace(sqrtm(sqrt_rho1 @ rho2 @ sqrt_rho1))
    fidelity = np.abs(fidelity) ** 2  # Ensuring numerical stability

    # Trace Distance: D = (1/2) * Tr(|rho1 - rho2|)
    trace_distance = 0.5 * np.trace(np.abs(sqrtm((rho1 - rho2).T @ (rho1 - rho2))))

    # Hilbert-Schmidt Distance: D_HS = Tr[(rho1 - rho2)^2]
    hs_distance = np.trace((rho1 - rho2) @ (rho1 - rho2))

    return {
        "Fidelity": fidelity,
        "Trace Distance": np.real(trace_distance),
        "Hilbert-Schmidt Distance": np.real(hs_distance)
    }


