import numpy as np
import librosa
import librosa.display
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt



# Load Audio File
def load_audio(file_path):
    """Load an audio file using SoundFile instead of Librosa"""
    audio, sr = sf.read(file_path)
    print(f"Loaded audio with {len(audio)} samples at {sr} Hz")
    return audio, sr

# Plot Waveform
def plot_waveform(audio, sr, title="Waveform"):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

# Plot Frequency Spectrum
def plot_spectrum(audio, sr, title="Frequency Spectrum"):
    freqs, psd = signal.welch(audio, sr, nperseg=1024)
    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs, psd)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.show()

# def downsample_audio(audio, sr, new_sr):
#     """Downsamples the audio to a lower sampling rate."""
#     factor = sr // new_sr
#     downsampled_audio = audio[::factor]
#     print(f"Downsampled from {sr} Hz to {new_sr} Hz")
#     return downsampled_audio, new_sr

# def lowpass_filter(audio, sr, cutoff=0.4):
#     """Applies a low-pass filter before downsampling to prevent aliasing."""
#     nyquist = sr / 2
#     cutoff_freq = cutoff * nyquist  # Define cutoff frequency
#     sos = signal.butter(4, cutoff_freq, btype='low', fs=sr, output='sos')
#     filtered_audio = signal.sosfilt(sos, audio)
#     print(f"Applied low-pass filter with cutoff at {cutoff_freq} Hz")
#     return filtered_audio

# Update main function to include low-pass filtering
def main():
    file_path = "sample.wav"  # Change this to your file path
    original_audio, sr = load_audio(file_path)

    # Plot Original Waveform and Spectrum
    plot_waveform(original_audio, sr, title="Original Waveform")
    plot_spectrum(original_audio, sr, title="Original Spectrum")

    # # Apply Low-Pass Filter
    # filtered_audio = lowpass_filter(original_audio, sr)

    # # Apply Downsampling
    # new_sr = sr // 4  # Reduce sampling rate (adjust as needed)
    # downsampled_audio, new_sr = downsample_audio(filtered_audio, sr, new_sr)

    # # Plot Processed Waveform and Spectrum
    # plot_waveform(downsampled_audio, new_sr, title="Filtered & Downsampled Waveform")
    # plot_spectrum(downsampled_audio, new_sr, title="Filtered & Downsampled Spectrum")

if __name__ == "__main__":
    main()
