import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import noisereduce as nr
import librosa
import time
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
                chunk = self.audio[int_pos:int_new_pos]
                outdata[:len(chunk), 0] = chunk
                outdata[len(chunk):, 0] = 0
            else:
                outdata[:, 0] = self.audio[int_pos:int_new_pos]
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

        self.noise_button = QPushButton("🔇 Reduce Noise", self)
        self.noise_button.clicked.connect(self.apply_noise_reduction)
        layout.addWidget(self.noise_button)

        self.playback_speed_label = QLabel("Playback Speed:", self)
        layout.addWidget(self.playback_speed_label)
        self.speed_combo = QComboBox(self)
        self.speed_combo.addItems(["0.25", "0.5", "0.75", "1", "1.25", "1.5", "2"])
        self.speed_combo.setCurrentText("1")
        layout.addWidget(self.speed_combo)

        self.echo_button = QPushButton("🎶 Add Echo", self)
        self.echo_button.clicked.connect(self.add_echo)
        layout.addWidget(self.echo_button)

        self.revert_button = QPushButton("↩️ Revert Audio", self)
        self.revert_button.clicked.connect(self.revert_audio)
        layout.addWidget(self.revert_button)

        self.play_button = QPushButton("⏯️ Play/Pause Audio", self)
        self.play_button.clicked.connect(self.toggle_play_pause)
        layout.addWidget(self.play_button)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.stop_button = QPushButton("⏹️ Stop Audio", self)
        self.stop_button.clicked.connect(self.stop_audio)
        layout.addWidget(self.stop_button)

        self.save_button = QPushButton("💾 Save Processed Audio", self)
        self.save_button.clicked.connect(self.save_audio)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def import_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open WAV File", "", "WAV files (*.wav)")
        if file_path:
            self.status_label.setText("🔄 Loading...")
            self.worker = LoadWavWorker(file_path)
            self.worker.update_status.connect(self.status_label.setText)
            self.worker.wav_loaded.connect(self.handle_wav_loaded)
            self.worker.start()

    def handle_wav_loaded(self, audio_data):
        global recorded_audio, processed_audio
        recorded_audio = audio_data
        processed_audio = recorded_audio  # Initial copy

    def record_audio(self):
        global recorded_audio
        try:
            self.status_label.setText("🎤 Recording...")
            recorded_audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
            sd.wait()
            recorded_audio = recorded_audio.flatten()
            self.status_label.setText("✅ Recording complete!")
        except Exception as e:
            self.status_label.setText(f"❌ Error: {e}")

    def apply_noise_reduction(self):
        global recorded_audio, processed_audio
        if recorded_audio is None:
            self.status_label.setText("⚠️ No audio recorded or imported!")
            return
        self.status_label.setText("🔧 Reducing noise...")
        processed_audio = nr.reduce_noise(y=recorded_audio, sr=SAMPLE_RATE)
        self.status_label.setText("✅ Noise reduction applied!")

    def add_echo(self):
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
        global recorded_audio, processed_audio
        if recorded_audio is None:
            self.status_label.setText("⚠️ No original audio to revert to!")
            return
        processed_audio = recorded_audio
        self.speed_combo.setCurrentText("1")
        self.status_label.setText("✅ Audio reverted to original!")

    def toggle_play_pause(self):
        global processed_audio
        if processed_audio is None:
            self.status_label.setText("⚠️ No processed audio available!")
            return
        if not hasattr(self, 'worker') or not self.worker.isRunning():
            self.status_label.setText("▶️ Playing audio...")
            self.progress_bar.setValue(0)
            speed = float(self.speed_combo.currentText())
            self.worker = PlayAudioWorker(processed_audio, speed=speed)
            self.worker.update_progress.connect(self.progress_bar.setValue)
            self.worker.playback_done.connect(self.playback_finished)
            self.worker.start()
        else:
            self.worker.toggle_pause()
            if self.worker.paused:
                self.status_label.setText("⏸️ Audio paused")
            else:
                self.status_label.setText("▶️ Audio resumed")

    def stop_audio(self):
        global is_playing
        if is_playing:
            sd.stop()
            is_playing = False
            self.progress_bar.setValue(0)
            self.status_label.setText("⏹️ Playback stopped")

    def playback_finished(self):
        self.status_label.setText("✅ Playback finished")

    def save_audio(self):
        global processed_audio
        if processed_audio is None:
            self.status_label.setText("⚠️ No processed audio available!")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "WAV files (*.wav)")
        if file_path:
            sf.write(file_path, processed_audio, SAMPLE_RATE)
            self.status_label.setText(f"💾 File saved: {file_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioApp()
    window.show()
    sys.exit(app.exec())
