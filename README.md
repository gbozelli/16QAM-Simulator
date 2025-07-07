# 16-QAM Communication System Simulation

This project simulates a digital communication system using 16-QAM (Quadrature Amplitude Modulation). It includes all the essential blocks: modulation, pulse shaping, DAC interpolation, channel modeling with AWGN, coherent demodulation, and symbol decision. The simulation is implemented in Python, with modular code organized into a reusable library.

## Features

- 16-QAM bit-to-symbol mapping and demapping
- Raised cosine pulse shaping (Nyquist filtering)
- Digital-to-analog conversion via zero-padding in the frequency domain
- I/Q modulation with a carrier signal
- AWGN channel model
- Coherent demodulation and filtering
- Bit Error Rate (BER) calculation
- Visualization tools for constellation, time-domain signal, and spectrum

## Project Structure

```
.
├── qam_lib.py          # Library with all system components
├── simulate_qam.py     # Example script to test the system
├── README.md           # Project documentation (this file)
```

## Dependencies

Make sure to install the required Python libraries:

```bash
pip install numpy matplotlib scipy
```

## How to Run the Simulation

Run the example script:

```bash
python simulate_qam.py
```

This will execute a full transmission and reception chain, print the BER value, and optionally show plots of the constellation, signal, and spectrum.

## Example Output

```
Estimated BER: 0.0200
```

## Parameters

You can tune the system behavior by modifying the parameters in `simulate_qam.py`:

```python
n_bits = 100         # Number of bits
sps = 4              # Samples per symbol
beta = 0.35          # Raised cosine roll-off
f_c = 1.5e6          # Carrier frequency (Hz)
sigma = 0.2          # AWGN noise standard deviation
t = 1e-6             # Symbol duration (s)
```

## Visualization

Use the plotting functions from the library for debugging and analysis:

```python
plotar_constelacao(symbols)               # Before transmission
plotar_constelacao(received_symbols)      # After reception
plotar_espectro(modulated_signal, fs, "Modulated Spectrum")
plotar_sinal_tempo(time, signal, "Time-domain Signal")
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Gabriel Bozelli Dias

