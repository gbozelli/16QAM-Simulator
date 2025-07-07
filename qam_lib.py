# qam_lib.py 

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq, ifft, ifftshift
import sys
import math

#Transmitter

def generate_binary_data(n_bits:int)->np.array:
    """
    Generate an array of bits in On-Off signalization
    Args:
        n_bits (int)
    Returns:
        np.array
    """
    return np.random.randint(0, 2, n_bits)

def map_16qam(bits:np.array)->np.array:
    """
    Generate an array of 16QAM coordinates using an array of bits.
    If the array don't have a length multiple of 4, bits zeros
    will be added to make the mapping possible.
    Args:
        bits (np.array): array of shape (n,)
    Returns:
        np.array: array of shape (n/4,4)
    """
    while len(bits)%4 != 0:
      print("bits: Lenght of array isn't multiple of 4")
      bits = np.append(bits,0)

    bits = bits.reshape(-1, 4)
    mapping = {
        '0000': [-3, 3], '0001': [-3, 1], '0010': [-3, -1], '0011': [-3, -3],
        '0100': [-1, 3], '0101': [-1, 1], '0110': [-1, -1], '0111': [-1, -3],
        '1000': [1, 3],  '1001': [1, 1],  '1010': [1, -1],  '1011': [1, -3],
        '1100': [3, 3],  '1101': [3, 1],  '1110': [3, -1],  '1111': [3, -3]
    }
    bit_strings = np.apply_along_axis(lambda x: ''.join(x.astype(str)), 1, bits)
    coordinates = np.array([mapping[s] for s in bit_strings])
    return coordinates

def upsampling(symbols:np.array, sps:int)->np.array:
    """
    Upsample an array of symbols with a defined number of
    samples per symbol.
    Args:
        symbols (np.array): array of shape (n,)
        sps (int): symbols per second
    Returns:
        np.array: array of shape (n*sps,)
    """
    if sps < 1:
      print("Samples per symbol is less than 1. The function doesn't do anything")
      return symbols
    upsampled_symbols = np.array([])
    for s in symbols:
        pulse = np.zeros(sps)
        pulse[0] = s
        upsampled_symbols = np.concatenate((upsampled_symbols, pulse))
    return upsampled_symbols

def raised_cosine_filter(beta:float, sps:int, num_taps:int)->np.array:
    """
    Generate the coeffcients of a raised cosine filter
    Args:
        beta (float): roll-off factor 
        sps (int): symbols per second
        num_taps (int): number of coefficients of filter
    Returns:
        np.array: array of shape (num_taps,)
    """
    if beta > 1:
      print("beta is greater than 1")
      sys.exit(1)
    t = np.arange(-num_taps // 2, num_taps // 2 + 1)
    t = t / sps
    h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2)
    return h

def nyquist_filter(symbols:np.array, h:np.array)->np.array:
    """
    Apply the 'h' filter, including the centering after the IIS.
    Args:
        symbols (np.array): array of symbols
        h (np.array): array of filter coefficients
    Returns:
        np.array: array of shape (len(symbols),)

    """
    signal = np.convolve(symbols, h, mode='same')
    return signal 

def digital_to_analog_converter(signal_I:np.array, signal_Q:np.array)->np.array:
    """
    DAC
    Args:
      signal_I (np.array): array of signal in-phase
      signal_Q (np.array): array of signal in-quadrature
    Returns:
      analog_signal_I (np.array)
      analog_signal_Q (np.array)
    """
    n = int(len(signal_I)/2)
    complex_signal = signal_I + 1j * signal_Q
    fft_complex = fftshift(fft(complex_signal))
    fft_complex_with_padding = np.concatenate([np.zeros(n), fft_complex, np.zeros(n)])
    descentralized_spectrum = ifftshift(fft_complex_with_padding)
    analog_signal = ifft(descentralized_spectrum)
    analog_signal_I = np.real(analog_signal)
    analog_signal_Q = np.imag(analog_signal)
    return analog_signal_I, analog_signal_Q

def single_digital_to_analog_converter(signal:np.array)->np.array:
    """
    DAC
    Args:
      signal_I (np.array): array of signal in-phase
      signal_Q (np.array): array of signal in-quadrature
    Returns:
      analog_signal_I (np.array)
      analog_signal_Q (np.array)
    """
    n = int(len(signal)/2)
    complex_signal = signal
    fft_complex = fftshift(fft(complex_signal))
    fft_complex_with_padding = np.concatenate([np.zeros(n), fft_complex, np.zeros(n)])
    descentralized_spectrum = ifftshift(fft_complex_with_padding)
    analog_signal = ifft(descentralized_spectrum)
    return analog_signal

def modulation(signal_I:np.array, signal_Q:np.array, f_c:float, t:float, sps:int)->np.array:
    """
    Modulation of both I and Q signal 
    Args:
      signal_I (np.array): analog signal in-phase
      signal_Q (np.array): analog signal in-quadrature
      f_c (float): frequency of carrier
      t (float): period of bit
      sps (int): samples per symbol
    Returns:
      Modulated signal (np.array)
    """ 
    n = (sps/2)
    time = np.arange(0, len(signal_I)*t/n, 4*t/(2*sps)) 
    if sps == 10:
      time = time[:-1]
    sinal_modulado_I = signal_I * np.cos(2 * np.pi * f_c * time)
    sinal_modulado_Q = signal_Q * np.sin(2 * np.pi * f_c * time)
    return sinal_modulado_I - sinal_modulado_Q

#Channel

def add_noise(modulated_signal:np.array, sigma:int)->np.array:
    """
    Adds noise in a AWGN channel
    Args:
      modulated_signal (np.array)
      sigma (int): sigma for noise generation
    Returns:
      transmitted_signal (np.array)
    """
    noise = np.random.normal(loc=0, scale=sigma, size=len(modulated_signal))
    return modulated_signal + noise

#Receiver

def demodulate_sinal(transmitted_signal:np.array, f_c:float, t:float, sps:int)->np.array:
    """
    Demodulate the signal and applies a low-pass filter
    Args:
      transmitted_signal (np.array): analog signal
      f_c (float): frequency of carrier
      t (float): period of bit
      sps (int): samples per symbol
    Returns:
      Signal_I and Signal_Q
    """
    n = sps/2
    time = np.arange(0, len(transmitted_signal)*t/n, 4*t/(2*sps))  
    if sps == 10:
      time = time[:-1]
    rec_I = transmitted_signal * np.cos(2 * np.pi * f_c * time)
    rec_Q = -transmitted_signal * np.sin(2 * np.pi * f_c * time)
    N = len(rec_I)
    complex_signal = rec_I + 1j * rec_Q
    fft_complex = fftshift(fft(complex_signal))
    for i in range(int(N/2.5)):
      fft_complex[i] = 0
      fft_complex[-i] = 0
    spectrum_descentralized = ifftshift(fft_complex)
    signal = ifft(spectrum_descentralized)
    signal_I = 4 * np.real(signal)
    signal_Q = 4 * np.imag(signal)
    return signal_I, signal_Q
  
def demap_16qam(symbols:np.array)->np.array:
    """
    Decides the signature of each symbol based on minimal distance.
    Args:
        symbols (np.array): array of shape (n,)
    Returns:
        np.array: array of shape (n/2,2)
    """
    I, Q = symbols
    levels = np.array([-3, -1, 1, 3])
    N = min(len(I), len(Q))
    I_approx = levels[np.abs(I[:N, None] - levels).argmin(axis=1)]
    Q_approx = levels[np.abs(Q[:N, None] - levels).argmin(axis=1)]
    return np.column_stack((I_approx, Q_approx))

#Analysis

def power_of_signal(signal:np.array)->float:
    """Calculate the power of a signal."""
    return np.mean(np.abs(signal)**2)

def ber(transmitted_symbols:np.array, original_symbols:np.array, n_bits:int, sps:int)->float:
    """Calculates the bit error ratio"""
    errors = 0
    n = min(len(transmitted_symbols), len(original_symbols))
    for i in range(n):
        if original_symbols[i, 0] != transmitted_symbols[i, 0] or original_symbols[i, 1] != transmitted_symbols[i, 1]:
            errors += 1
    return errors / n_bits
