"""Kits for measuring photon state tomography data.

This module provides tools for processing and visualizing quantum optical measurement data,
including demodulation, temporal mode matching, and histogram visualization.

classes
==========
    `Histogram`: A 2D histogram for visualizing complex-valued measurement data.
    `Demodulator`: Optimized demodulation processor with preset parameters.
    `TemporalModeMatcher`: Handles temporal mode matching operations.
"""

from supportkit import generate_complex_2dcoord
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from warnings import warn
from math import isclose
from copy import copy

__all__ = [
    'Histogram',
    'Demodulator',
    'TemporalModeMatcher',
]

class Histogram:
    """A 2D histogram for visualizing complex-valued measurement data.

    Attributes:
        D (np.ndarray): 2D array storing the histogram counts.
        S (np.ndarray): Complex coordinates for the histogram.

    Methods:
        `accumulate_to_histogram`: Accumulate measurement data to histogram. S = X + iP
        `get_normalized_histogram`: Return histogram with maxima value be 1.
        `plot`: Plot normalized histogram.

    Example usage:
    >>> import numpy as np
    >>> from measurekit import Histogram
    >>> # make histogram
    >>> his = Histogram(n_row_col=101, max_x_p=5.0)
    >>> # accumulate data
    >>> for _ in range(2**16):
    >>>     x = np.random.normal(0, 1)
    >>>     p = np.random.normal(0, 1)
    >>>     s = x + 1j*p
    >>>     his.accumulate_to_histogram(s)
    >>> # plot and get data
    >>> D, S = his.get_normalized_histogram(), his.S
    >>> his.plot("Random Gaussian Samples in Phase Space")
    """
    def __init__(self, n_row_col: int, max_x_p: float):
        """
        Args:
            n_row_col (int): number of rows and columns of histogram
            max_x_p (float): maxima value of X, P, where S = X + iP.
        """
        self.D = np.zeros([n_row_col, n_row_col])
        self.n_row_col = n_row_col
        self.max_x_p = max_x_p
        self.bin_size = 2 * max_x_p / n_row_col
        self.S = generate_complex_2dcoord(max_x_p, n_row_col)

    def _s2ij(self, s: complex):
        """Find closest coordinate S = X + iP belongs to. None for exceed."""
        x = s.real
        p = s.imag
        if abs(x) > self.max_x_p or abs(p) > self.max_x_p:
            return None, None
        i = int((p + self.max_x_p) / self.bin_size)
        j = int((x + self.max_x_p) / self.bin_size)
        return i, j

    def get_normalized_histogram(self):
        """Return histogram with maxima value be 1."""
        max_val = np.max(self.D)
        return self.D / max_val if max_val != 0 else self.D

    def accumulate_to_histogram(self, measured_s: complex):
        """Accumulate measurement data to histogram. S = X + iP."""
        i, j = self._s2ij(measured_s)
        if i is not None:
            self.D[i, j] += 1

    def plot(self, title="Histogram", cmap='hot', ax=None):
        """Plot normalized histogram.

        Example usage:
        >>> fig, axs = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
        >>> his_s.plot(title="Signal Histogram)", ax=axs[0])
        >>> his_h.plot(title="Noise Histogram", ax=axs[1])
        >>> plt.tight_layout()
        >>> plt.show()
        """
        data = self.get_normalized_histogram()
        extent = [-self.max_x_p, self.max_x_p, -self.max_x_p, self.max_x_p]

        # Use current axis if none provided
        if ax is None:
            ax = plt.gca()

        im = ax.imshow(data, extent=extent, origin='lower', cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('P')
        ax.set_aspect('equal')  # keep square pixels

        # Only add colorbar if using default plotting
        if ax is plt.gca():
            plt.colorbar(im, ax=ax, label='Normalized counts')

        return im  # return image object for optional colorbar usage


class Demodulator:
    """Optimized demodulation processor with preset parameters.
    
    For A(t) is baseband signal that mixed with a carrier with frequency fc, forming the signal x(t).
    Demodulation can be seen as a process to obtain A(t) for a given x(t) and fc.
    And we use ADC to sample our signal x(t) into x[n], and this class perform demodulation task.

    Methods:
        `print_setting`: Print settings about demodulation.
        `hilbert_ssb_demod`: Demod using Hilbert transform.
        `fast_shift_demod`: Demod with duplicate shifted signal, to approximate Hilbert transform.
        `iq_demod`: Fast IQ demodulation with FIR filtering (optimized for known fixed fs and fc).
        `low_pass_filter`: apply FIR low-pass filter to the input signal using the configured taps.

    Example usage:
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from measurekit import Demodulator
    >>> # make x(t) = A(t) * carrier
    >>> fc, fs = 50e6, 1e9
    >>> t = np.arange(1000) / fs
    >>> baseband = np.exp(t/2e-7)
    >>> carrier = np.cos(2*np.pi*fc*t)
    >>> signal = baseband * carrier
    >>> # perform demod from x(t) to obtain A(t)
    >>> de = Demodulator(fc=fc, fs=fs, n_samples=len(signal))
    >>> demoded = de.fast_shift_demod(signal) # should equal to baseband
    >>> # plot
    >>> plt.plot(np.abs(demoded), label='demoded')
    >>> plt.plot(baseband, '--', label='baseband')
    >>> plt.plot(signal, label='signal', alpha=0.5)
    >>> plt.legend()
    >>> plt.show()
    
    """
    def __init__(self, fc, fs, n_samples, cutoff=20e6, num_taps=31):
        """
        Args:
            fc (float): carrier frequency to be demoded out of signal.
            fs (float): sampling rate of the digitized signal.
            n_samples (int): length of digitized signal.
            cutoff (float): cutoff freq for low pass filter operation
            num_taps (int): Number of FIR filter taps (odd number recommended).
        """
        # general
        self.fc = fc
        self.fs = fs
        self.n_samples = n_samples
        self.t = np.arange(n_samples) * 1/fs
        self.cpx_lo = np.exp(-1j*2*np.pi*fc*self.t)

        # for fast_shift_demod
        self.quad_shift_pts = int(fs / fc / 4)
        if not isclose(fs / fc / 4, self.quad_shift_pts, rel_tol=0, abs_tol=1e-9):
            warn("fs is not interger mutiple of 4*fc, fast shift demodulation will not work.")

        # for iq_demod (simple LPF: sinc * window)
        self.cutoff = cutoff
        self.num_taps = num_taps
        nyq = self.fs / 2
        norm_cutoff = cutoff / nyq
        taps = np.sinc(2 * norm_cutoff * (np.arange(num_taps) - (num_taps - 1) / 2))
        window = np.hamming(num_taps)
        fir = taps * window
        self.fir = fir / np.sum(fir)

    def print_setting(self):
        """print settings about demodulation"""
        string = (
            f"fc (carrier frequency) : {self.fc*1e-6:.3f} MHz\n"
            f"fs (sampling rate) : {self.fs*1e-9:.3f} GHz\n"
            f"n_samples : {self.n_samples} pts\n"
            f"cutoff : {self.cutoff*1e-6:.3f} MHz\n"
            f"num_taps : {self.num_taps}\n"
        )
        print(string)

    def hilbert_ssb_demod(self, signal):
        "Demod using Hilbert transform."
        analytic = hilbert(signal)
        demoded = analytic * self.cpx_lo
        return demoded

    def fast_shift_demod(self, signal, second_order=True):
        "Demod with duplicate shifted signal, to approximate Hilbert transform."
        # forward shift
        pi_half_shifted = np.zeros_like(signal)
        pi_half_shifted[self.quad_shift_pts:] = signal[:-self.quad_shift_pts]
        if not second_order: 
            return (signal + 1j*pi_half_shifted) * self.cpx_lo

        # backward shift
        nag_pi_half_shifted = np.zeros_like(signal)
        nag_pi_half_shifted[:-self.quad_shift_pts] = signal[self.quad_shift_pts:]
        cpx_signal = signal + 1j*(pi_half_shifted - nag_pi_half_shifted)/2
        return cpx_signal * self.cpx_lo


    def iq_demod(self, signal):
        "Fast IQ demodulation with FIR filtering (optimized for known fixed fs and fc)."
        baseband = signal * self.cpx_lo
        # Apply FIR filter
        filtered = np.convolve(baseband, self.fir, mode='same')  # center-aligned
        return filtered * 2

    def low_pass_filter(self, signal):
        """Apply FIR low-pass filter to the input signal using the configured taps."""
        return np.convolve(signal, self.fir, mode='same')  # center-aligned
    
class TemporalModeMatcher:
    """Handles temporal mode matching operations.

    For qubit emmision be a(t), a time-depedent mode, we may define a time-indepdenet mode 'a', s.t.
    a = int f(t) a(t) dt,
    to descrilbe qubit state. f(t) is said to be the filter function, the choice of it will influence
    the time-indepdenet mode 'a', and it is often choose to maximize SNR.

    As advised in [eth-6886-02], p.50, we can use averaged emmision data as filter.
    For single shot measurement, qubit might emit a photon at different time, so we use convolution
    to find best matching for every shot.

    Methods:
        `regist_filter`: Regist a filter, it will normalize it for you.
        `pad_or_trim_filter`: Pad zero or trim some points from the registed filter and return it.
        `plot_tmm_info`: Print and plot temporal mode matching convolution window.
        `perform_tmm`: Perform temporal mode matching with complex signal.
    
    Example usage:
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from measurekit import TemporalModeMatcher
    >>> # make a baseband signal and one with noise
    >>> fc, fs = 50e6, 1e9
    >>> t = np.arange(1000) / fs
    >>> baseband = np.concatenate([np.exp(t/2e-7), np.zeros(200)]) / 1e8
    >>> def add_noise(signal, sigma=1e-7):
    >>>     noise = np.random.normal(
    >>>         loc=0.0, 
    >>>         scale=sigma, 
    >>>         size=signal.shape
    >>>     )
    >>>     return signal + noise
    >>> signal = add_noise(baseband)
    >>> # set filter to be the averaged signal
    >>> tmm = TemporalModeMatcher(fs=fs)
    >>> tmm.regist_filter(baseband)
    >>> tmm.regist_filter(tmm.pad_or_trim_filter())
    >>> tmm.plot_tmm_info(signal)
    >>> # perform tmm to signal that is with noise
    >>> s = tmm.perform_tmm(signal)
    >>> print('tmm result:', s)
    """
    def __init__(self, fs):
        """Set sampling rate.

        Example usage:
        >>> tmmer = TemporalModeMatcher(fs=1e9)
        >>> tmmer.regist_filter(avg_sig)
        >>> tmmer.regist_filter(tmmer.pad_or_trim_filter())
        >>> tmmer.plot_tmm_info(signal)
        """
        self.fs = fs

    def regist_filter(self, digitized_filter):
        """Regist a filter, it will normalize it for you.
        
        The normalization goes "int sum |f[n]|^2 / fs = 1".
        """
        norm_factor = np.sqrt(np.sum(np.abs(digitized_filter)**2) / self.fs)
        self.digitized_filter = digitized_filter / norm_factor

    def pad_or_trim_filter(self, pad_front=-40, pad_end=-40):
        """Pad zero or trim some points from the registed filter and return it.
        
        Example usage:
        >>> tmmer.regist_filter(avg_sig)
        >>> tmmer.regist_filter(tmmer.pad_or_trim_filter())
        >>> tmmer.plot_tmm_info(signal)
        """
        trimed = copy(self.digitized_filter)

        # Front padding or trimming
        if pad_front > 0:
            trimed = np.concatenate([np.zeros(pad_front), trimed])
        elif pad_front < 0:
            trimed = trimed[-pad_front:]  # Remove from front

        # End padding or trimming
        if pad_end > 0:
            trimed = np.concatenate([trimed, np.zeros(pad_end)])
        elif pad_end < 0:
            trimed = trimed[:pad_end]  # Remove from end

        return trimed

    def plot_tmm_info(self, signal):
        """Print and plot temporal mode matching convolution window.

        Example usage:
        >>> tmmer.regist_filter(avg_sig)
        >>> tmmer.regist_filter(tmmer.pad_or_trim_filter())
        >>> tmmer.plot_tmm_info(signal)
        """
        n_inprod = len(signal) - len(self.digitized_filter) + 1
        if n_inprod <= 0:
            print(f'Not valid for length of filter is larger then that of signal.'
                  f' ({len(self.digitized_filter)} > {len(signal)})')
            return 
        print('# of inner product: ', n_inprod)

        # rescale filter to make the plot easy to see
        factor = np.max(np.abs(signal)) / np.max(self.digitized_filter)

        plt.plot(np.abs(signal), label='abs(signal)', color='orange', alpha=0.6)
        plt.plot(factor*self.digitized_filter, label='filter start', color='blue')
        plt.plot(factor*np.concatenate([np.zeros(n_inprod-1), self.digitized_filter]), 
                label='filter end', color='red')
        plt.legend()
        plt.show()

    def perform_tmm(self, signal, index=None):
        """Perform temporal mode matching with complex signal.

        Arg:
            index(int): if None, matching, otherwise use it to do inner product.
        
        1. Use abs(signal) and filter to find the best alignment.
        2. Return the complex inner product at that best-matching position.
        """
        if index is None:
            # match magnitudes to find best alignment
            correlation_result = np.correlate(
                np.abs(signal), self.digitized_filter, mode='valid'
            )
            best_idx = np.argmax(correlation_result)
        else:
            best_idx = index

        # compute complex inner product for best aligment case
        matched_segment = signal[best_idx : best_idx + len(self.digitized_filter)]
        inner_product = np.dot(self.digitized_filter, matched_segment)
        return inner_product
