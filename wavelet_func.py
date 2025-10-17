import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt     # pip install PyWavelets



def wavelet_full_analysis(df, col, wavelet='cmor1.5-1.0', scales=np.arange(1, 128), sampling_rate=1.0):
    """
    Performs continuous wavelet transform to extract power, phase, and time-frequency representation.

    Parameters:
        df (pd.DataFrame): DataFrame with time series data.
        col (str): Target column name.
        wavelet (str): Complex wavelet (e.g., 'cmor1.5-1.0', 'morlet').
        scales (np.ndarray): Array of wavelet scales to analyze.
        sampling_rate (float): Sampling rate in Hz (e.g. 1/86400 for daily).

    Returns:
        dict with keys: power, phase, scalogram, scales, frequencies
    """

    df = df.copy()
    df['diff'] = df[col].diff().bfill()

    signal = df[col].values
    signal_diff = df['diff'].values

    dt = 1.0 / sampling_rate

    # Continuous Wavelet Transform
    coefficients, freqs = pywt.cwt(signal, scales=scales, wavelet=wavelet, sampling_period=dt)
    coefficients_diff, freqs_diff = pywt.cwt(signal_diff, scales=scales, wavelet=wavelet, sampling_period=dt)

    # Power Calculations
    power = np.abs(coefficients) ** 2
    power_db = 10 * np.log10(power + 1e-10)  # Avoid log(0) issues   POwer spectrum in  DB

    gamma = 0.01  # between 0 (strong compression) and 1 (no compression)
    power_gamma = power ** gamma  # gamma = 0.5 → square root || gamma = 0.3 → softer than sqrt but not as compressed as log
    power_sqrt = np.sqrt(power)
    power_softlog = np.log1p(power)  # ≈ log(1 + Sxx)
    power_asinh = np.arcsinh(power)
    power_log2 = np.log2(power + 1e-10)

    # getting phase
    phase = np.angle(coefficients)

    # Power Calculations for diff
    power_diff = np.abs(coefficients_diff) ** 2
    power_db_diff = 10 * np.log10(power_diff + 1e-10)  # Avoid log(0) issues   POwer spectrum in  DB

    gamma_diff = 0.01  # between 0 (strong compression) and 1 (no compression)
    power_gamma_diff = power_diff ** gamma_diff  # gamma = 0.5 → square root || gamma = 0.3 → softer than sqrt but not as compressed as log
    power_sqrt_diff = np.sqrt(power_diff)
    power_softlog_diff = np.log1p(power_diff)  # ≈ log(1 + Sxx)
    power_asinh_diff = np.arcsinh(power_diff)
    power_log2_diff = np.log2(power_diff + 1e-10)

    # getting phase for diff
    phase_diff = np.angle(coefficients_diff)

    """
    'viridis'	Default, green to purple
    'plasma'	Bright purple to yellow
    'magma'	Black to white-ish
    'cividis'	Colorblind-friendly
    """
    # Plot: Scalogram (Spectrogram)
    # print("\n🎛️ Wavelet Scalogram (Spectrogram):")

    # plt.figure(figsize=(20, 4))
    # plt.imshow(power, extent=[0, len(signal), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
    # plt.title(f"Wavelet Scalogram (Power Spectrum Over Time | raw power) {col}")
    # plt.xlabel("Time (Samples)")
    # plt.ylabel("Frequency [Hz]")
    # plt.colorbar(label="Power")
    # plt.tight_layout()
    # plt.show()

    if True:
        plt.figure(figsize=(12.3, 3))
        plt.plot(df.index, df[col], label=f'{col}')
        plt.xlabel('Time')
        plt.title(f'{col} Original vs Signal Amplitude')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.margins(x=0)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if True:
        plt.figure(figsize=(20, 4))
        plt.imshow(power_sqrt, extent=[0, len(signal), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.title(f"Wavelet Scalogram (Power Spectrum Over Time | power_sqrt) {col}")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="Power power_sqrt")
        plt.tight_layout()
        plt.show()

    if True:
        plt.figure(figsize=(20, 4))
        plt.imshow(power_sqrt_diff, extent=[0, len(signal_diff), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.title(f"Wavelet Scalogram (Power Spectrum Over Time | power_sqrt) {col} diff")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="Power power_sqrt")
        plt.tight_layout()
        plt.show()

    if False:
        plt.figure(figsize=(20, 4))
        plt.imshow(power_softlog, extent=[0, len(signal), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.title(f"Wavelet Scalogram (Power Spectrum Over Time | power_softlog) {col}")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="Power power_softlog")
        plt.tight_layout()
        plt.show()

    if False:
        plt.figure(figsize=(20, 4))
        plt.imshow(power_softlog_diff, extent=[0, len(signal_diff), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.title(f"Wavelet Scalogram (Power Spectrum Over Time | power_softlog) {col} diff")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="Power power_softlog")
        plt.tight_layout()
        plt.show()

    if True:
        power_asinh_2 = power_asinh ** (1 / 4)
        plt.figure(figsize=(20, 4))
        plt.imshow(power_asinh_2, extent=[0, len(signal_diff), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.title(f"Wavelet Scalogram (Power Spectrum Over Time | power_asinh_2 TEST) {col}")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency")
        plt.colorbar(label="Power power_asinh")
        plt.tight_layout()
        plt.show()

        power_asinh_2_diff = power_asinh_diff ** (1 / 4)
        plt.figure(figsize=(20, 4))
        plt.imshow(power_asinh_2_diff, extent=[0, len(signal_diff), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.title(f"Wavelet Scalogram (Power Spectrum Over Time | power_asinh_2 TEST) {col} Diff")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency")
        plt.colorbar(label="Power power_asinh")
        plt.tight_layout()
        plt.show()

    if False:
        plt.figure(figsize=(20, 4))
        plt.imshow(power_asinh, extent=[0, len(signal), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.title(f"Wavelet Scalogram (Power Spectrum Over Time | power_asinh ) {col}")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency")
        plt.colorbar(label="Power power_asinh")
        plt.tight_layout()
        plt.show()

    if False:
        plt.figure(figsize=(20, 4))
        plt.imshow(power_asinh_diff, extent=[0, len(signal_diff), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.title(f"Wavelet Scalogram (Power Spectrum Over Time | power_asinh) {col} diff")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency")
        plt.colorbar(label="Power power_asinh")
        plt.tight_layout()
        plt.show()

    if True:
        plt.figure(figsize=(12.3, 3))
        plt.plot(df.index, df[col], label=f'{col}')
        plt.xlabel('Time')
        plt.title(f'{col} Original vs Signal Amplitude')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.margins(x=0)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if False:
        plt.figure(figsize=(20, 4))
        plt.imshow(power_gamma, extent=[0, len(signal), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.title(f"Wavelet Scalogram (Power Spectrum Over Time | power_gamma) {col}")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="Power power_gamma")
        plt.tight_layout()
        plt.show()

    if False:
        plt.figure(figsize=(20, 4))
        plt.imshow(power_gamma_diff, extent=[0, len(signal_diff), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.title(f"Wavelet Scalogram (Power Spectrum Over Time | power_gamma) {col} diff")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="Power power_gamma")
        plt.tight_layout()
        plt.show()

    if False:
        plt.figure(figsize=(20, 4))
        plt.imshow(power_log2, extent=[0, len(signal), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.title(f"Wavelet Scalogram (Power Spectrum Over Time | power_log2) {col}")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="Power power_log2")
        plt.tight_layout()
        plt.show()

    if False:
        plt.figure(figsize=(20, 4))
        plt.imshow(power_log2_diff, extent=[0, len(signal_diff), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.title(f"Wavelet Scalogram (Power Spectrum Over Time | power_log2) {col} diff")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="Power power_log2")
        plt.tight_layout()
        plt.show()

    if False:
        plt.figure(figsize=(20, 4))
        plt.imshow(power_db, extent=[0, len(signal), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.colorbar(label="Power [dB]")
        plt.title(f"Wavelet Scalogram (Power Spectrum in [DB] Over Time)  {col}")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency [Hz]")
        plt.tight_layout()
        plt.show()

    if False:
        plt.figure(figsize=(20, 4))
        plt.imshow(power_db_diff, extent=[0, len(signal_diff), freqs[-1], freqs[0]], aspect='auto', cmap='plasma')
        plt.colorbar(label="Power [dB]")
        plt.title(f"Wavelet Scalogram (Power Spectrum in [DB] Over Time)  {col} diff")
        plt.xlabel("Time (Samples)")
        plt.ylabel("Frequency [Hz]")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12.3, 3))
        plt.plot(df.index, df[col], label=f'{col}')
        plt.xlabel('Time')
        plt.title(f'{col} Original vs Signal Amplitude')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.margins(x=0)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if False:
        # Plot: Mean Power Spectrum
        mean_power = power.mean(axis=1)
        plt.figure(figsize=(20, 4))
        plt.plot(freqs, mean_power)
        plt.title("Mean Power Spectrum by Frequency")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Average Power")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot: Phase Spectrum (at middle sample)
        mid = signal.shape[0] // 2
        plt.figure(figsize=(20, 4))
        plt.plot(freqs, phase[:, mid])
        plt.title("Phase Spectrum at Midpoint")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Phase [radians]")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        'power': power,
        'phase': phase,
        'scales': scales,
        'frequencies': freqs
    }
