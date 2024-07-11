from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QFrame, QDialog, QLineEdit, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QFont, QFontDatabase
from PyQt5.QtCore import Qt
import sys
import os
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
matplotlib.use('Qt5Agg', force=True)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def parse_rttm(filename, segment_start_time):
    segments = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            start = float(parts[3]) + segment_start_time
            duration = float(parts[4])
            label = parts[7]
            segments.append({"start": start, "duration": duration, "label": label})
    return segments
    
def adjust_alpha(color, alpha_factor=0.8):
    r, g, b, a = color
    a = max(a * alpha_factor, 0.0)
    return (r, g, b, a)

def plot_waveform(y, sr, segments, start_time):
    # Generate a colormap with a large number of colors
    num_labels = len(set(segment["label"] for segment in segments))
    colormap = plt.get_cmap('tab20', num_labels)  

    # Create a dictionary to map each label to a unique color
    labels = list(set(segment["label"] for segment in segments))
    colors = {label: colormap(i / num_labels) for i, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(14, 5))

    for segment in segments:
        start_sample = int(segment["start"] * sr)
        end_sample = int((segment["start"] + segment["duration"]) * sr)

        if end_sample > len(y):
            end_sample = len(y)

        label = segment["label"]
        color = colors[label]

        times = librosa.samples_to_time(range(start_sample, end_sample), sr=sr) + start_time
        ax.plot(times, y[start_sample:end_sample], color=color, alpha=0.8, label=f'{label}')

    full_audio_times = librosa.samples_to_time(range(len(y)), sr=sr) + start_time
    ax.plot(full_audio_times, y, color='gray', alpha=0.2, label='Full Audio')

    ax.set_title('Waveform')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

    # Create a legend with unique labels
    legend_handles = [plt.Line2D([0], [0], color=colors[label], lw=4, label=f'{label}') for label in colors]
    ax.legend(handles=legend_handles, loc='upper right')

    return fig


def split_audio_and_rttm(y, sr, rttm_lines, start_time, duration):
    # Calculate the total duration of the audio in seconds
    total_duration = librosa.get_duration(y=y, sr=sr)

    # Calculate the end time of the segment, ensuring it does not exceed the total duration
    end_time = min(start_time + duration, total_duration)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract the audio segment
    segment = y[start_sample:end_sample]

    # Initialize an empty list to store the RTTM segments
    segments = []
    for line in rttm_lines:
        parts = line.strip().split()
        start_time_rttm = float(parts[3])
        duration_rttm = float(parts[4])
        end_time_rttm = start_time_rttm + duration_rttm

        # Check if the RTTM segment overlaps with the desired segment
        if start_time_rttm < end_time and end_time_rttm > start_time:
            # Calculate the new start time and duration for the segment within the desired time range
            new_start_time = max(0, start_time_rttm - start_time)
            new_duration = min(duration_rttm, end_time - start_time_rttm)

            # Append the segment to the list
            segments.append({
                "start": new_start_time,
                "duration": new_duration,
                "label": parts[7]  # Assuming the label is the 8th part in the RTTM line
            })

    return segment, sr, segments


class CustomInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Times")
        self.setMinimumWidth(300)
        self.setStyleSheet("background-color: black;")

        self.start_label = QLabel("Start Time (seconds):")
        self.start_label.setStyleSheet("color: white;")
        self.start_edit = QLineEdit()
        self.start_edit.setPlaceholderText("Enter start time")
        self.start_edit.setStyleSheet(
            "color: black; background-color: #E0FFFF; font-family: Verdana; font-size: 8pt; font-style: italic;")

        self.end_label = QLabel("End Time (seconds):")
        self.end_label.setStyleSheet("color: white;")
        self.end_edit = QLineEdit()
        self.end_edit.setPlaceholderText("Enter end time")
        self.end_edit.setStyleSheet(
            "color: black; background-color: #E0FFFF; font-family: Verdana; font-size: 8pt; font-style: italic;")

        self.ok_button = QPushButton("OK")
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #99ccff; 
                color: black; 
                border: 2px solid #99ccff;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #80bfff;
            }
            QPushButton:pressed {
                background-color: #6699ff;
            }
        """)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #99ccff; 
                color: black; 
                border: 2px solid #99ccff;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #80bfff;
            }
            QPushButton:pressed {
                background-color: #6699ff;
            }
        """)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout = QVBoxLayout()
        layout.addWidget(self.start_label)
        layout.addWidget(self.start_edit)
        layout.addWidget(self.end_label)
        layout.addWidget(self.end_edit)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.ok_button.clicked.connect(self.validate_and_accept)
        self.cancel_button.clicked.connect(self.reject)

        # Connect text changed signal to update placeholder styles
        self.start_edit.textChanged.connect(self.update_placeholder_style)
        self.end_edit.textChanged.connect(self.update_placeholder_style)

    def update_placeholder_style(self):
        if self.start_edit.text().strip():
            self.start_edit.setStyleSheet(
                "color: black; background-color: #E0FFFF; font-family: Verdana; font-size: 8pt;")
        else:
            self.start_edit.setStyleSheet(
                "color: black; background-color: #E0FFFF; font-family: Verdana; font-size: 8pt; font-style: italic;")

        if self.end_edit.text().strip():
            self.end_edit.setStyleSheet(
                "color: black; background-color: #E0FFFF; font-family: Verdana; font-size: 8pt;")
        else:
            self.end_edit.setStyleSheet(
                "color: black; background-color: #E0FFFF; font-family: Verdana; font-size: 8pt; font-style: italic;")

    def validate_and_accept(self):
        start_time_text = self.start_edit.text().strip()
        end_time_text = self.end_edit.text().strip()

        if not start_time_text:
            self.start_edit.setPlaceholderText("Enter start time (required)")
            self.start_edit.setStyleSheet(
                "color: red; background-color: #E0FFFF; font-family: Verdana; font-size: 8pt; font-style: italic;")
        else:
            self.start_edit.setStyleSheet(
                "color: black; background-color: #E0FFFF; font-family: Verdana; font-size: 8pt;")

        if not end_time_text:
            self.end_edit.setPlaceholderText("Enter end time (required)")
            self.end_edit.setStyleSheet(
                "color: red; background-color: #E0FFFF; font-family: Verdana; font-size: 8pt; font-style: italic;")
        else:
            self.end_edit.setStyleSheet(
                "color: black; background-color: #E0FFFF; font-family: Verdana; font-size: 8pt;")

        if start_time_text and end_time_text:
            self.accept()

    def get_times(self):
        start_time = float(self.start_edit.text()) if self.start_edit.text() else 0.0
        end_time = float(self.end_edit.text()) if self.end_edit.text() else 0.0
        return start_time, end_time

class CustomWarningDialog(QDialog):
    def __init__(self, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Warning")
        self.setMinimumWidth(400)
        self.setStyleSheet("background-color: black; color: white;")

        # Main layout
        main_layout = QVBoxLayout(self)

        # Message and icon layout
        message_layout = QHBoxLayout()

        # Warning icon
        self.icon_label = QLabel()
        self.icon_label.setPixmap(
            QPixmap("warning.png").scaled(80, 80,
                                                                                                          Qt.KeepAspectRatio))
        message_layout.addWidget(self.icon_label)

        # Message
        self.message_label = QLabel(message)
        self.message_label.setStyleSheet("font-family: 'Arial'; font-size: 9pt; color: white;")
        self.message_label.setWordWrap(True)
        message_layout.addWidget(self.message_label)

        main_layout.addLayout(message_layout)

        # OK button
        self.ok_button = QPushButton("Proceed")
        self.ok_button.setFixedSize(80, 30)  
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #99ccff; 
                color: black; 
                border: 2px solid #99ccff;
                border-radius: 15px;
                font-family: 'Verdana'; 
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #80bfff;
            }
            QPushButton:pressed {
                background-color: #6699ff;
            }
        """)
        main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

        self.ok_button.clicked.connect(self.accept)

        self.adjustSize()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.filename_wav = None
        self.filename_rttm = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Waveform Visualization')
        self.setGeometry(100, 100, 1280, 832)
        self.setStyleSheet("background-color: #F5F5F5;")

        self.bg_label = QLabel(self)
        pixmap = QPixmap('Main.png')  
        self.bg_label.setPixmap(pixmap)
        self.bg_label.setScaledContents(True)
        self.bg_label.setGeometry(0, 0, 1280, 832)

        font_id = QFontDatabase.addApplicationFont("MountainsofChristmas-Regular.ttf")
        font_family = QFontDatabase.applicationFontFamilies(font_id)[0]

        self.title_label = QLabel("Language\nDiarization", self)
        title_font = QFont(font_family, 70)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: black;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setGeometry(650, 50, 600, 300)

        self.upload_audio_button = QPushButton("Upload the Audio File", self)
        self.upload_audio_button.setFont(QFont("Arial", 12))
        self.upload_audio_button.setStyleSheet("""
            QPushButton {
                background-color: #99ccff; 
                color: black; 
                border: 2px solid #99ccff;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #80bfff;
            }
            QPushButton:pressed {
                background-color: #6699ff;
            }
        """)
        self.upload_audio_button.setGeometry(100, 100, 300, 50)
        self.upload_audio_button.clicked.connect(self.upload_audio_file)

        self.selected_audio_label = QLabel("Selected Audio File: [file_name]", self)
        self.selected_audio_label.setFont(QFont("Arial", 10))
        self.selected_audio_label.setStyleSheet("color: black;")
        self.selected_audio_label.setGeometry(100, 160, 300, 20)
        self.selected_audio_label.hide()

        self.upload_rttm_button = QPushButton("Upload the RTTM File", self)
        self.upload_rttm_button.setFont(QFont("Arial", 12))
        self.upload_rttm_button.setStyleSheet("""
            QPushButton {
                background-color: #99ccff; 
                color: black; 
                border: 2px solid #99ccff;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #80bfff;
            }
            QPushButton:pressed {
                background-color: #6699ff;
            }
        """)
        self.upload_rttm_button.setGeometry(100, 200, 300, 50)
        self.upload_rttm_button.clicked.connect(self.upload_rttm_file)

        self.selected_rttm_label = QLabel("Selected RTTM File: [file_name]", self)
        self.selected_rttm_label.setFont(QFont("Arial", 10))
        self.selected_rttm_label.setStyleSheet("color: black;")
        self.selected_rttm_label.setGeometry(100, 260, 300, 20)
        self.selected_rttm_label.hide()

        self.detect_button = QPushButton("Display Waveform", self)
        self.detect_button.setFont(QFont("Arial", 12))
        self.detect_button.setStyleSheet("""
            QPushButton {
                background-color: #99ccff; 
                color: black; 
                border: 2px solid #99ccff;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #80bfff;
            }
            QPushButton:pressed {
                background-color: #6699ff;
            }
        """)
        self.detect_button.setGeometry(100, 300, 300, 50)
        self.detect_button.clicked.connect(self.prompt_times)

        self.warning_label = QLabel("Please upload the Audio File and RTTM File", self)
        self.warning_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.warning_label.setStyleSheet("color: red;")
        self.warning_label.setGeometry(100, 360, 300, 30)
        self.warning_label.hide()

        self.main_frame = QFrame(self)
        self.main_frame.setStyleSheet("background-color: #ededf3; border-radius: 15px;")
        self.main_frame.setGeometry(100, 450, 1080, 400)

        self.reset_button = QPushButton("Reset", self)
        self.reset_button.setFont(QFont("Arial", 8))
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #ff6666; 
                color: black; 
                border: 2px solid #ff6666;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #ff9999;
            }
            QPushButton:pressed {
                background-color: #cc3333;
            }
        """)
        self.reset_button.setGeometry(1080, 370, 80, 30)
        self.reset_button.clicked.connect(self.reset_ui)
        self.reset_button.hide()  # Initially hidden

        self.show()

    def resizeEvent(self, event):
        # Override resizeEvent to handle resizing of elements
        self.updateUI()

    def updateUI(self):
        # Update positions and sizes of UI elements based on current window size
        self.bg_label.setGeometry(0, 0, self.width(), self.height())

        
        title_width = 600  # Adjust width 
        title_height = 300  # Adjust height 
        title_x = self.width() - title_width - 50  # Adjust the padding from the right edge
        title_y = 50  # Adjust vertical position if needed
        self.title_label.setGeometry(title_x, title_y, title_width, title_height)
        
        # Position reset button to the rightmost side
        self.reset_button.setGeometry(self.width() - 180, 420 - 50, 60, 30)

        # Update main frame to cover the whole area
        main_frame_x = 100
        main_frame_y = 420
        main_frame_width = self.width() - 200
        main_frame_height = self.height() - 500
        self.main_frame.setGeometry(main_frame_x, main_frame_y, main_frame_width, main_frame_height)

        # Update other widget positions and sizes as 

    def upload_audio_file(self):
        options = QFileDialog.Options()
        self.filename_wav, _ = QFileDialog.getOpenFileName(self, "Select WAV file", "", "WAV files (*.wav)", options=options)
        if self.filename_wav:
            self.selected_audio_label.setText(f"Selected Audio File: {os.path.basename(self.filename_wav)}")
            self.selected_audio_label.show()
            self.warning_label.hide()

    def upload_rttm_file(self):
        options = QFileDialog.Options()
        self.filename_rttm, _ = QFileDialog.getOpenFileName(self, "Select RTTM file", "", "RTTM files (*.rttm)", options=options)
        if self.filename_rttm:
            self.selected_rttm_label.setText(f"Selected RTTM File: {os.path.basename(self.filename_rttm)}")
            self.selected_rttm_label.show()
            self.warning_label.hide()

    def prompt_times(self):
        if not self.filename_wav or not self.filename_rttm:
            self.warning_label.show()
            return

        dialog = CustomInputDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            start_time, end_time = dialog.get_times()
            self.detect_languages(start_time, end_time)

    def detect_languages(self, start_time=None, end_time=None):
        y, sr = librosa.load(self.filename_wav, sr=None)

        with open(self.filename_rttm, 'r') as file:
            rttm_lines = file.readlines()

        if start_time is not None and start_time > librosa.get_duration(y=y, sr=sr):
            warning_dialog = CustomWarningDialog("Entered start time exceeds the duration of the whole audio file.\nPlease enter a valid start time.")
            warning_dialog.exec_()
            return

        if end_time is not None and end_time > librosa.get_duration(y=y, sr=sr):
            warning_dialog = CustomWarningDialog(f"Entered end time exceeds the duration of the whole audio file.\nWaveform will be plotted from {start_time} to the end of the audio file.")
            warning_dialog.exec_()
            end_time = librosa.get_duration(y=y, sr=sr)

        if start_time is not None and end_time is not None:
            duration = end_time - start_time
            y, sr, segments = split_audio_and_rttm(y, sr, rttm_lines, start_time, duration)
        else:
            segments = parse_rttm(self.filename_rttm, 0.0)

        fig = plot_waveform(y, sr, segments, start_time)

        layout = self.main_frame.layout()
        if layout is not None:
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().setParent(None)

        canvas = FigureCanvas(fig)
        new_layout = QVBoxLayout()
        new_layout.addWidget(canvas)
        self.main_frame.setLayout(new_layout)

        self.reset_button.show()  # Show reset button after plotting

    def reset_ui(self):
    # Delete the main frame and recreate it
        self.main_frame.deleteLater()  # Remove the existing main frame
        self.main_frame = QFrame(self)  # Recreate the main frame
        main_frame_x = 100
        main_frame_y = 420
        main_frame_width = self.width() - 200
        main_frame_height = self.height() - 500
        self.main_frame.setStyleSheet("background-color: #ededf3; border-radius: 15px;")
        self.main_frame.setGeometry(main_frame_x, main_frame_y, main_frame_width, main_frame_height)
        self.main_frame.show()  # Show the newly created main frame
    
    # Hide the reset button
        self.reset_button.hide()
    
    # Hide selected file labels
        self.selected_audio_label.hide()
        self.selected_rttm_label.hide()
    
    # Reset filenames
        self.filename_wav = None
        self.filename_rttm = None

    # Hide warning label if visible
        self.warning_label.hide()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
