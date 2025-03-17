import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import noisereduce as nr
import librosa
import time
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QSlider, QProgressBar, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

SAMPLE_RATE = 44100
DURATION = 5
recorded_audio = None
processed_audio = None
is_playing = False


# Worker Thread to Import WAV Without Freezing UI
class LoadWavWorker(QThread):
    update_status = pyqtSignal(str)
    wav_loaded = pyqtSignal(np.ndarray)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        self.update_status.emit("📂 Loading WAV file...")
        try:
            audio, sr = librosa.load(self.file_path, sr=SAMPLE_RATE, mono=True)
            self.wav_loaded.emit(audio)
            self.update_status.emit("✅ WAV File Imported Successfully!")
        except Exception as e:
            self.update_status.emit(f"❌ Error: {e}")


class PlayAudioWorker(QThread):
    update_progress = pyqtSignal(int)
    playback_done = pyqtSignal()

    def __init__(self, audio, speed=1.0):
        super().__init__()
        self.audio = audio
        self.speed = speed
        self.paused = False
        self.index = 0.0

    def toggle_pause(self):
        self.paused = not self.paused

    def run(self):
        global is_playing
        is_playing = True
        if self.speed != 1.0:
            new_length = int(len(self.audio) / self.speed)
            self.audio = np.interp(np.linspace(0, len(self.audio), new_length), np.arange(len(self.audio)), self.audio)
            self.speed = 1.0
        blocksize = 1024

        def callback(outdata, frames, time_info, status):
            if self.paused:
                outdata.fill(0)
                return
            int_pos = int(self.index)
            int_new_pos = int_pos + frames
            # Handle boundary condition
            if int_new_pos > len(self.audio):
                int_new_pos = len(self.audio)
                chunk = self.audio[int_pos:int_new_pos].flatten()  # Ensure it's 1D
                outdata[:len(chunk), 0] = chunk
                outdata[len(chunk):, 0] = 0  # Zero-pad if needed
            else:
                chunk = self.audio[int_pos:int_new_pos].flatten()  # Flatten here as well
                outdata[:, 0] = chunk
            self.index = int_new_pos

        stream = sd.OutputStream(channels=1, samplerate=SAMPLE_RATE, callback=callback, blocksize=blocksize)
        with stream:
            while self.index < len(self.audio):
                if not is_playing:
                    self.update_progress.emit(0)
                    return
                progress = int((self.index / len(self.audio)) * 100)
                self.update_progress.emit(progress)
                time.sleep(0.05)
            sd.sleep(100)  # Allow stream to flush out remaining audio
        is_playing = False
        self.playback_done.emit()


class AudioApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Processing App")
        self.setGeometry(100, 100, 400, 550)

        layout = QVBoxLayout()

        self.status_label = QLabel("🎙️ Ready to record or import", self)
        layout.addWidget(self.status_label)

        self.import_button = QPushButton("📂 Import WAV File", self)
        self.import_button.clicked.connect(self.import_audio)
        layout.addWidget(self.import_button)

        self.record_button = QPushButton("🎤 Record Audio", self)
        self.record_button.clicked.connect(self.record_audio)
        layout.addWidget(self.record_button)

        self.playback_speed_label = QLabel("Playback Speed:", self)
        layout.addWidget(self.playback_speed_label)
        self.speed_combo = QComboBox(self)
        self.speed_combo.addItems(["0.25", "0.5", "0.75", "1", "1.25", "1.5", "2"])
        self.speed_combo.setCurrentText("1")
        layout.addWidget(self.speed_combo)
    
        self.noise_button = QPushButton("🔇 Reduce Noise", self)
        self.noise_button.clicked.connect(self.apply_noise_reduction)
        layout.addWidget(self.noise_button)

        self.echo_button = QPushButton("🎶 Add Echo", self)
        self.echo_button.clicked.connect(self.add_echo)
        layout.addWidget(self.echo_button)

        self.revert_button = QPushButton("↩️ Revert Audio", self)
        self.revert_button.clicked.connect(self.revert_audio)
        layout.addWidget(self.revert_button)

        self.play_button = QPushButton("⏯️ Play/Pause Audio", self)
        self.play_button.clicked.connect(self.toggle_play_pause)
        layout.addWidget(self.play_button)

        self.freq_button = QPushButton("📊 Analyze Frequency", self)
        self.freq_button.clicked.connect(self.analyze_frequency)
        layout.addWidget(self.freq_button)

        self.progress_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(100)
        self.progress_slider.setValue(0)
        self.progress_slider.sliderReleased.connect(self.slider_released)
        self.progress_slider.sliderPressed.connect(self.slider_pressed)
        layout.addWidget(self.progress_slider)

        self.stop_button = QPushButton("⏹️ Stop Audio", self)
        self.stop_button.clicked.connect(self.stop_audio)
        layout.addWidget(self.stop_button)

        self.save_button = QPushButton("💾 Save Processed Audio", self)
        self.save_button.clicked.connect(self.save_audio)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def import_audio(self):
        """
        Opens a file dialog to import a WAV file.
        - Clears previous audio.
        - Loads the WAV file asynchronously to prevent UI freezing.
        - Updates the status label with progress.
        """
        global recorded_audio, processed_audio
        file_path, _ = QFileDialog.getOpenFileName(self, "Open WAV File", "", "WAV files (*.wav)")
        if file_path:
            self.status_label.setText("🔄 Loading...")
            
            # Clear previous audio
            recorded_audio = None
            processed_audio = None

            self.worker = LoadWavWorker(file_path)
            self.worker.update_status.connect(self.status_label.setText)
            self.worker.wav_loaded.connect(self.handle_wav_loaded)
            self.worker.start()

    def handle_wav_loaded(self, audio_data):
        """
        Handles the loaded WAV file.
        - Stores the imported audio in recorded_audio and processed_audio.
        """
        global recorded_audio, processed_audio
        recorded_audio = audio_data
        processed_audio = recorded_audio  # Initial copy

    def record_audio(self):
        """
        Handles recording audio.
        - Starts/stops recording when the button is clicked.
        - Uses a buffer to store recorded audio dynamically.
        - Stops the recording stream properly.
        - Updates UI elements accordingly.
        """
        global recorded_audio, processed_audio
        if hasattr(self, 'recording') and self.recording:
            self.rec_stream.stop()
            self.rec_stream.close()
            self.recording = False
            self.record_button.setText("🎤 Record Audio")
            self.status_label.setText("✅ Recording stopped!")

            # Compute actual duration based on start time
            self.record_duration = time.time() - self.start_time
            recorded_audio = self.record_buffer[:int(self.record_duration * SAMPLE_RATE)].copy()
            processed_audio = recorded_audio  # Store for playback

        else:
            try:
                self.status_label.setText("🎤 Recording...")
                self.record_button.setText("⏹️ Stop Recording")
                self.recording = True
                
                # Clear previously imported or recorded audio
                recorded_audio = None
                processed_audio = None
                
                # Ensure a valid input device
                input_device = sd.default.device[0]
                if input_device is None:
                    self.status_label.setText("❌ No audio input device found!")
                    self.record_button.setText("🎤 Record Audio")
                    return
                
                # Capture start time for dynamic recording length
                self.start_time = time.time()

                # Buffer to store recorded audio
                self.record_buffer = np.zeros((60 * SAMPLE_RATE,), dtype='float32')  # Up to 60s buffer
                self.record_index = 0  # Track how much data is recorded
                
                # Function to continuously store recorded data
                def callback(indata, frames, time, status):
                    if status:
                        print(status)
                    end_idx = self.record_index + frames
                    if end_idx > len(self.record_buffer):
                        end_idx = len(self.record_buffer)
                    self.record_buffer[self.record_index:end_idx] = indata[:end_idx - self.record_index, 0]
                    self.record_index = end_idx

                # Start recording stream
                self.rec_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=callback)
                self.rec_stream.start()
                
                self.status_label.setText("🎤 Recording... Click stop to finish.")
            except Exception as e:
                self.status_label.setText(f"❌ Error: {e}")
                self.record_button.setText("🎤 Record Audio")

    def apply_noise_reduction(self):
        """
        Applies noise reduction to the processed audio.
        - Uses the `noisereduce` library to clean up noise.
        - Updates the status label after processing.
        """
        global recorded_audio, processed_audio
        if recorded_audio is None:
            self.status_label.setText("⚠️ No audio recorded or imported!")
            return
        self.status_label.setText("🔧 Reducing noise...")
        processed_audio = nr.reduce_noise(y=recorded_audio, sr=SAMPLE_RATE)
        self.status_label.setText("✅ Noise reduction applied!")

    def add_echo(self):
        """
        Adds an echo effect to the processed audio.
        - Delays the signal and overlays it with a reduced amplitude.
        - Updates the status label after applying the effect.
        """
        global processed_audio
        if processed_audio is None:
            self.status_label.setText("⚠️ No processed audio available!")
            return
        self.status_label.setText("🎶 Adding Echo effect...")
        delay_samples = int(0.2 * SAMPLE_RATE)
        echo_audio = np.pad(processed_audio, (0, delay_samples), mode='constant', constant_values=0)
        echo_audio[delay_samples:] += processed_audio * 0.5
        processed_audio = echo_audio
        self.status_label.setText("✅ Echo effect added!")

    def revert_audio(self):
        """
        Reverts the processed audio back to the original recorded/imported audio.
        - Restores the original data from recorded_audio.
        - Resets the speed setting.
        """
        global recorded_audio, processed_audio
        if recorded_audio is None:
            self.status_label.setText("⚠️ No original audio to revert to!")
            return
        processed_audio = recorded_audio
        self.speed_combo.setCurrentText("1")
        self.status_label.setText("✅ Audio reverted to original!")

    def toggle_play_pause(self):
        """
        Handles play/pause functionality.
        - Selects the latest available audio (processed or recorded).
        - Starts playback using a separate worker thread.
        - Toggles pause if already playing.
        """
        global recorded_audio, processed_audio
        if processed_audio is not None:
            audio_to_play = processed_audio
        elif recorded_audio is not None:
            audio_to_play = recorded_audio
        else:
            self.status_label.setText("⚠️ No recorded or imported audio available!")
            return

        if not hasattr(self, 'worker') or not self.worker.isRunning():
            self.status_label.setText("▶️ Playing audio...")
            self.progress_slider.setValue(0)
            speed = float(self.speed_combo.currentText())
            self.worker = PlayAudioWorker(audio_to_play, speed=speed)
            self.worker.update_progress.connect(self.update_slider)
            self.worker.playback_done.connect(self.playback_finished)
            self.worker.start()
        else:
            self.worker.toggle_pause()
            if self.worker.paused:
                self.status_label.setText("⏸️ Audio paused")
            else:
                self.status_label.setText("▶️ Audio resumed")

    def stop_audio(self):
        """
        Stops audio playback immediately.
        - Stops the active stream and resets playback progress.
        """
        global is_playing
        if is_playing:
            sd.stop()
            is_playing = False
            self.progress_slider.setValue(0)
            self.status_label.setText("⏹️ Playback stopped")

    def playback_finished(self):
        """
        Updates the UI when playback is finished.
        """
        self.status_label.setText("✅ Playback finished")

    def slider_released(self):
        """
        Handles seeking in audio playback.
        - Adjusts playback position when the progress slider is moved.
        """
        # Update playback position based on the slider value
        if hasattr(self, 'worker') and self.worker.isRunning():
            new_index = int((self.progress_slider.value() / 100.0) * len(self.worker.audio))
            self.worker.index = new_index
            print(f"Seeked to {self.worker.index} samples.")
        self.seeking = False

    def slider_pressed(self):
        """
        Temporarily disables slider updates while seeking.
        """
        self.seeking = True

    def update_slider(self, progress):
        """
        Updates the progress slider during playback.
        - Prevents updates when the user is manually seeking.
        """
        if not getattr(self, 'seeking', False):
            self.progress_slider.setValue(progress)

    def save_audio(self):
        """
        Saves the processed audio to a WAV file.
        - Opens a save dialog and writes the processed audio.
        """
        global processed_audio
        if processed_audio is None:
            self.status_label.setText("⚠️ No processed audio available!")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "WAV files (*.wav)")
        if file_path:
            sf.write(file_path, processed_audio, SAMPLE_RATE)
            self.status_label.setText(f"💾 File saved: {file_path}")

    def analyze_frequency(self):
        """
        Performs frequency analysis of the original and processed audio.
        - Computes FFT to visualize frequency components.
        - Displays a comparison plot of both signals.
        """
        global recorded_audio, processed_audio
        if recorded_audio is None:
            self.status_label.setText("⚠️ No original audio available!")
            return

        self.status_label.setText("📊 Analyzing frequency...")
        
        # Compute FFT for original audio
        N_original = len(recorded_audio)
        freq_original = np.fft.fftfreq(N_original, d=1/SAMPLE_RATE)
        spectrum_original = np.abs(np.fft.fft(recorded_audio))

        freq_processed, spectrum_processed = None, None
        if processed_audio is not None and len(processed_audio) > 0:
            N_processed = len(processed_audio)
            freq_processed = np.fft.fftfreq(N_processed, d=1/SAMPLE_RATE)
            spectrum_processed = np.abs(np.fft.fft(processed_audio))

        # Plot the frequency spectrum
        plt.figure(figsize=(10, 5))
        plt.plot(freq_original[:N_original//2], spectrum_original[:N_original//2], label="Original Audio", linestyle='dashed')

        if freq_processed is not None:
            plt.plot(freq_processed[:N_processed//2], spectrum_processed[:N_processed//2], label="Processed Audio")

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Frequency Spectrum Comparison")
        plt.legend()
        plt.grid()
        plt.show()

        self.status_label.setText("✅ Frequency analysis completed!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioApp()
    window.show()
    sys.exit(app.exec())
