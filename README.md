# Creative Speaker Detection

A real-time audio-visual speaker detection system that combines computer vision and speech processing to accurately identify and track speakers in video streams.

## Features

- Real-time speaker diarization using WhisperX
- Advanced face detection and tracking
- Lip movement analysis for speaker verification
- WebRTC support for live video streaming
- Temporal correlation between audio and visual cues
- Debug audio logging capabilities
- Confidence scoring for speaker-face mapping

## Requirements

- Python 3.7+
- PyTorch
- WhisperX
- FastAPI
- aiortc (for WebRTC)
- CUDA-capable GPU (recommended)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/creative-speaker.git
cd creative-speaker

# Install dependencies
pip install -r requirements.txt
```

## Usage
Starting the WebRTC Server
```bash
python webrtc.py
```

This will start the WebRTC server on http://localhost:8000.

### Using the Speaker Detection System

```python
from audio_visual_speaker_detection import AudioVisualSpeakerDetector

# Initialize the detector
detector = AudioVisualSpeakerDetector(
    device="cuda",
    model_name="large-v2",
    min_speakers=1,
    max_speakers=4
)

# Process video segments
segments, face_detections = detector.process_video_segment(audio, video_frames)
```

## Configuration
The system can be configured with various parameters:
```
device: Computing device ("cuda" or "cpu")

model_name: WhisperX model size

compute_type: Computation precision ("float16" or "float32")

min_speakers: Minimum number of speakers to detect

max_speakers: Maximum number of speakers to detect

confidence_threshold: Threshold for speaker-face mapping (default: 0.7)
```