import numpy as np
import matplotlib.pyplot as plt
from qam_lib import *

# System parameters
n_bits = 100
sps = 4
beta = 0.35
num_taps = n_bits
t = 1e-6
f_c = 1.5e6
sigma = 0.2

# Generate binary data
bits = generate_binary_data(n_bits)

# Map bits to 16QAM symbols
symbols = map_16qam(bits)  # shape: (n_bits/4, 2)
symbols_I = symbols[:, 0]
symbols_Q = symbols[:, 1]

# Upsample I and Q symbols
upsampled_I = upsampling(symbols_I, sps)
upsampled_Q = upsampling(symbols_Q, sps)

# Generate raised cosine filter
h = raised_cosine_filter(beta, sps, num_taps)

# Apply Nyquist pulse shaping filter
shaped_I = nyquist_filter(upsampled_I, h)
shaped_Q = nyquist_filter(upsampled_Q, h)

# Digital-to-analog conversion
analog_I, analog_Q = digital_to_analog_converter(shaped_I, shaped_Q)

# Modulate I and Q signals onto carrier
modulated_signal = modulation(analog_I, analog_Q, f_c, t)

# Transmit signal through AWGN channel
received_signal = add_noise(modulated_signal, sigma)

# Demodulate received signal
received_I, received_Q = demodulate_sinal(received_signal, f_c, t)

# Make decisions based on minimum distance
received_symbols = demap_16qam((received_I, received_Q))

# Calculate bit error rate
ber_val = ber(received_symbols, symbols, n_bits, sps)
print(f"Estimated BER: {ber_val:.4f}")
