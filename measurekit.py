"""Kits for the measurment of photon state tomography data.

ref: [eth-6886-02]

This module provides tools to do tomography measurement, 
including demodulation, temporal mode matching, and histogram visualization.
The measurement may goes like:
1. Measure an averaged emission, by the help of `Demodulator`.
2. Use averaged emission to build filter function, regist into `TemporalModeMatcher`.
3. Perform many single shot measurement, each time accumulate the count into `Histogram` object.

classes
==========
    `Histogram`: A 2D histogram for visualizing complex-valued measurement data.
    `Demodulator`: Optimized demodulation processor with preset parameters.
    `TemporalModeMatcher`: Handles temporal mode matching operations.
"""

from .supportkit import generate_complex_2dcoord
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from warnings import warn
from math import isclose
from copy import copy
import csv


__all__ = [
    'Histogram',
    'Demodulator',
    'TemporalModeMatcher',
    'cpx_to_rphi',
]

class Histogram:
    """A 2D histogram for visualizing complex-valued measurement data.

    Attributes:
        D (np.ndarray): 2D array storing the histogram counts.
        S (np.ndarray): Complex coordinates for the histogram.
        comment (str): a string that is added as common.
        
    Methods:
        `accumulate_to_histogram`: Accumulate measurement data to histogram. S = X + iP
        `get_normalized_histogram`: Return histogram with maxima value be 1.
        `plot`: Plot normalized histogram.
        `save_to_csv`: Save histogram to csv, also include comment.
    
    Static method:
        `read_from_csv`: Read histogram from csv, return Histogram object.

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
    def __init__(self, n_row_col: int, max_x_p: float, comment=''):
        """
        Args:
            n_row_col (int): number of rows and columns of histogram
            max_x_p (float): maxima value of X, P, where S = X + iP.
        """
        self.D = np.zeros([n_row_col, n_row_col], dtype=int)
        self.n_row_col = n_row_col
        self.max_x_p = max_x_p
        self.bin_size = 2 * max_x_p / n_row_col
        self.S = generate_complex_2dcoord(max_x_p, n_row_col)
        self.comment = comment

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

    def resolution_down(self, factor: int):
        """Return a new Histogram object with reduced resolution.

        The number of rows and columns in the new histogram is determined by
        dividing the current number of rows/columns by 2**(factor-1).

        Args:
            factor (int): An integer >= 1.
                - factor=1: No change in resolution (divides by 2^0=1).
                - factor=2: Resolution halved (divides by 2^1=2).
                - factor=3: Resolution quartered (divides by 2^2=4), etc.

        Returns:
            Histogram: A new Histogram instance with the reduced resolution and aggregated data.

        Raises:
            ValueError: If factor is less than 1, or if the resulting
                        number of rows/columns would be less than 1, or if
                        the current n_row_col is not divisible by the reduction step.
        """
        if not isinstance(factor, int) or factor < 1:
            raise ValueError("factor must be a positive integer (>= 1).")

        if factor == 1:
            reduction_step = 1
        else:
            reduction_step = 2**(factor - 1)

        if self.n_row_col % reduction_step != 0:
            # This might occur if n_row_col is not a power of 2, and factor implies
            # a reduction_step (which is a power of 2) that doesn't evenly divide n_row_col.
            raise ValueError(
                f"Current n_row_col ({self.n_row_col}) is not divisible by "
                f"the calculated reduction step ({reduction_step}) derived from factor {factor}."
            )

        new_n_row_col = self.n_row_col // reduction_step

        if new_n_row_col < 1:
            raise ValueError(
                f"Factor {factor} results in a new_n_row_col ({new_n_row_col}) that is less than 1. "
                f"The reduction is too large for the current histogram size."
            )

        # Create the new histogram object. Its __init__ will generate the new S coordinates.
        new_hist = Histogram(n_row_col=new_n_row_col, max_x_p=self.max_x_p)

        # Populate the D matrix of the new histogram by summing counts from blocks
        # in the original histogram's D matrix.
        for i_new in range(new_n_row_col):
            for j_new in range(new_n_row_col):
                # Define the block in the old histogram corresponding to the new bin
                start_i_old = i_new * reduction_step
                end_i_old = start_i_old + reduction_step
                
                start_j_old = j_new * reduction_step
                end_j_old = start_j_old + reduction_step
                
                # Sum the counts within this block from the original D matrix
                block_sum = np.sum(self.D[start_i_old:end_i_old, start_j_old:end_j_old])
                new_hist.D[i_new, j_new] = block_sum
                
        return new_hist

    def save_to_csv(self, filepath: str):
        """Save histogram data, configuration, and comment to a CSV file."""
        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)

            # Write metadata
            writer.writerow(['n_row_col', 'max_x_p', 'comment'])
            writer.writerow([self.n_row_col, self.max_x_p, self.comment])

            # Write header for data section
            writer.writerow(['D'])

            # Write histogram data
            for row in self.D:
                writer.writerow(row)


    @staticmethod
    def read_from_csv(filepath: str):
        """Read histogram object from a CSV file (including metadata and comment)."""
        with open(filepath, mode='r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Extract metadata
        header = rows[0]
        values = rows[1]
        n_row_col = int(values[0])
        max_x_p = float(values[1])
        comment = values[2] if len(values) > 2 else ''

        # Create new Histogram
        hist = Histogram(n_row_col=n_row_col, max_x_p=max_x_p, comment=comment)

        # Parse histogram data
        d_data = rows[3:]  # Skip metadata and 'D' label row
        for i, row in enumerate(d_data):
            hist.D[i, :] = np.array(row, dtype=int)

        return hist


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
        self.demod_buffer = np.zeros(n_samples, dtype=np.complex128)

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

    ## Devlog: not faster then using for loop, although I don't know why
    # def fast_shift_demod_batch(self, signal, second_order=True, copy=True):
    #     """Demod a batch of signals with approximated Hilbert transform. See docstring of `fast_shift_demod`.
        
    #     Args:
    #         signal (2d np.array): Signal for many traces. If 1d, promote to 2d with 1 trace.
    #         second_order (bool): use second order approximation, see docsting.
    #     Return:
    #         demodulated (2d np.array): n-th elem is n-th demodulated trace.
    #     """
    #     # load signal into buffer, no allocation needed
    #     self.batch_demod_buffer.real[:, :] = signal # inplace copy of real part
    #     self.batch_demod_buffer.imag[:, :] = 0      # clear imaginary part

    #     if not second_order:
    #         # first order approximation, a(t) = x(t) + j*f(t)
    #         self.batch_demod_buffer.imag[:, self.quad_shift_pts:] = signal[:, :-self.quad_shift_pts]
    #     else:
    #         # second order approximation, a(t) = x(t) + j*[f(t) - b(t)] / 2
    #         self.batch_demod_buffer.imag[:, self.quad_shift_pts:]  =  0.5 * signal[:, :-self.quad_shift_pts]
    #         self.batch_demod_buffer.imag[:, :-self.quad_shift_pts] -= 0.5 * signal[:, self.quad_shift_pts:]

    #     # now buf is analytic signal a(t), demod by a(t) * exp(-j omega t)        
    #     if copy:
    #         return (self.batch_demod_buffer * self.cpx_lo[None, :])  # new array allocated here
    #     else:
    #         # in-place mutiplication, no allocation
    #         self.batch_demod_buffer *= self.cpx_lo[None, :]
    #         return self.batch_demod_buffer
    

    def fast_shift_demod(self, signal, second_order=True, copy=True):
        """Demod with duplicate shifted signal, to approximate Hilbert transform.
        
        Explanation:
            For x(t) = A(t)*cos(2pi fc t), the Hilber transform of it is u(t) = A(t)*sin(2pi fc t).
            If we shift the signal by pi/2 (d) in time, we have 
                f(t) = +A(t-d)*sin(2pi fc t), (FORARED FIRST ORDER APPROXIMATION).
            We can shift backward also, to have
                b(t) = -A(t+d)*sin(2pi fc t), (BACKWARD FIRST ORDER APPROXIMATION).
            Then, the avarged one will be a lot better, that is
                s(t) = [f(t) - b(t)] / 2, (SECOND ORDER APPROXIMATION).
            Then we can build approximated analytic signal by a(t) = x(t) + j*s(t).
        """
        # load signal into buffer, no allocation needed
        self.demod_buffer.real[:] = signal    # inplace copy of real part
        self.demod_buffer.imag[:] = 0         # clear imaginary part

        if not second_order:
            # first order approximation, a(t) = x(t) + j*f(t)
            self.demod_buffer.imag[self.quad_shift_pts:] = signal[:-self.quad_shift_pts]
        else:
            # second order approximation, a(t) = x(t) + j*[f(t) - b(t)] / 2
            self.demod_buffer.imag[self.quad_shift_pts:]  =  0.5 * signal[:-self.quad_shift_pts]
            self.demod_buffer.imag[:-self.quad_shift_pts] -= 0.5 * signal[self.quad_shift_pts:]

        # now buf is analytic signal a(t), demod by a(t) * exp(-j omega t)        
        if copy:
            return (self.demod_buffer * self.cpx_lo)  # new array allocated here
        else:
            # in-place mutiplication, no allocation
            self.demod_buffer *= self.cpx_lo
            return self.demod_buffer


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
        >>> tmm = TemporalModeMatcher(fs=1e9)
        >>> tmm.regist_filter(avg_sig)
        >>> tmm.regist_filter(tmmer.pad_or_trim_filter())
        >>> tmm.plot_tmm_info(signal)
        """
        self.fs = fs

    def regist_filter(self, mm_filter):
        """Regist a mode matching filter, it will normalize it for you.
        
        The normalization goes "int sum( |f[n]|^2  dt ) = 1, dt = 1/fs".
        """
        norm_factor = np.sqrt(np.sum(np.abs(mm_filter)**2) / self.fs)
        self.mm_filter = mm_filter / norm_factor
        self.mm_filter = np.ascontiguousarray(self.mm_filter)
        self.filter_len = len(self.mm_filter)

    def pad_or_trim_filter(self, pad_front=-40, pad_end=-40):
        """Pad zero or trim some points from the registed filter and return it.
        
        Example usage:
        >>> tmmer.regist_filter(avg_sig)
        >>> tmmer.regist_filter(tmmer.pad_or_trim_filter())
        >>> tmmer.plot_tmm_info(signal)
        """
        trimed = copy(self.mm_filter)

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
        n_inprod = len(signal) - len(self.mm_filter) + 1
        if n_inprod <= 0:
            print(f'Not valid for length of filter is larger then that of signal.'
                  f' ({len(self.mm_filter)} > {len(signal)})')
            return 
        print('# of inner product: ', n_inprod)

        # rescale filter to make the plot easy to see
        factor = np.max(np.abs(signal)) / np.max(self.mm_filter)

        plt.plot(np.abs(signal), label='abs(signal)', color='orange', alpha=0.6)
        plt.plot(factor*self.mm_filter, label='filter start', color='blue')
        plt.plot(factor*np.concatenate([np.zeros(n_inprod-1), self.mm_filter]), 
                label='filter end', color='red')
        plt.legend()
        plt.show()


    def perform_tmm(self, signal, index: int=None):
        """Perform temporal mode matching with complex signal.

        1. Use abs(signal) and filter to find the best alignment.
        2. Return the complex inner product at that best-matching position.

        Args:
            signal (1d numpy array): the complex signal to be matching.
            index (int): if None, matching, otherwise use it to do inner product.
        Returns:
            tuple:
                - inner_products (complex): temporal moda matching result.
                - best_indices (int): the matching index.
        """
        if index is None:
            # match magnitudes to find best alignment
            correlation_result = np.correlate(
                np.abs(signal), self.mm_filter, mode='valid'
            )
            best_idx = np.argmax(correlation_result)
        else:
            best_idx = index

        # compute complex inner product for best aligment case
        matched_segment = signal[best_idx : best_idx + self.filter_len]
        inner_product = np.dot(self.mm_filter, matched_segment) / self.fs
        return inner_product, best_idx


def cpx_to_rphi(complex_numbers):
    """Return polar coordinate (r, phi) for complex number(s)."""
    r = np.abs(complex_numbers)
    phi = np.angle(np.mean(complex_numbers))
    return r, phi
