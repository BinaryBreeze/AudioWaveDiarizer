# AudioWaveDiarizer
AudioWaveDiarizer is a PyQt5-based application designed to visualize audio waveforms and diarization segments from RTTM files. This tool allows users to upload audio files and RTTM files, specify time segments, and display the corresponding waveform with diarization labels.
## Features

- **Upload Audio Files**: Upload WAV files to visualize their waveforms.
- **Upload RTTM Files**: Upload RTTM files for diarization segment visualization.
- **Segment Visualization**: Specify start and end times to focus on specific segments of the audio.
- **Interactive Plotting**: Display audio waveforms with diarization labels using Matplotlib.
## Installation

- **Clone the Repository**:
   ```sh
   git clone https://github.com/BinaryBreeze/AudioWaveDiarizer.git
   cd AudioWaveDiarizer
- **Create a Virtual Environment**:
  ```sh
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
- **Install Dependencies**:
  ```sh
  pip install -r requirements.txt
## Usage

- Run the application with the following command:
  ```sh
  python main.py
1. **Upload Files**: Click on "Upload the Audio File" and "Upload the RTTM File" buttons to select your WAV and RTTM files respectively.
2. **Specify Time Segment**: Enter the start and end times (in seconds) to visualize a specific segment of the audio.
3. **Display Waveform**: Click the "Display Waveform" button to generate the waveform plot with diarization segments.
4. **Reset**: Use the "Reset" button to clear the interface and upload new files.
## Dependencies

- Python 3.7+
- PyQt5
- Librosa
- SoundFile
- Matplotlib
## License

This project is licensed under the MIT License. SSee the [LICENSE](LICENSE) file for details..

