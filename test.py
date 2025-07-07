import numpy as np
import matplotlib.pyplot as plt
from qam_lib import *

# --- 1. Parâmetros do sistema ---
n_bits = 100
sps = 4
beta = 0.35
num_taps = n_bits
t = 1e-6
f_c = 1.5e6
sigma = 0.2

# --- 2. Gerar dados binários ---
bits = generate_binary_data(n_bits)

# --- 3. Mapear bits para símbolos 16QAM ---
symbols = map_16qam(bits)  # shape: (n_bits/4, 2)
symbols_I = symbols[:, 0]
symbols_Q = symbols[:, 1]

# --- 4. Sobreamostragem ---
upsampled_I = upsampling(symbols_I, sps)
upsampled_Q = upsampling(symbols_Q, sps)

# --- 5. Filtro de cosseno levantado ---
h = raised_cosine_filter(beta, sps, num_taps)

# --- 6. Formatação de pulso (Nyquist) ---
shaped_I = nyquist_filter(upsampled_I, h)
shaped_Q = nyquist_filter(upsampled_Q, h)

# --- 7. Conversão D/A ---
analog_I, analog_Q = digital_to_analog_converter(shaped_I, shaped_Q)

# --- 8. Modulação ---
modulated_signal = modulation(analog_I, analog_Q, f_c, t)

# --- 9. Canal com ruído AWGN ---
received_signal = add_noise(modulated_signal, sigma)

# --- 10. Demodulação ---
received_I, received_Q = demodulate_sinal(received_signal, f_c, t)

# --- 11. Decisão (demapeamento 16QAM) ---
received_symbols = demap_16qam((received_I, received_Q))

# --- 12. Cálculo do BER ---
ber_val = ber(received_symbols, symbols, n_bits, sps)
print(f"BER estimado: {ber_val:.4f}")

