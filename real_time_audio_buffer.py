from datetime import datetime
import os
import tempfile
from typing import Optional
import wave
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeAudioBuffer:
    def __init__(self, max_duration=5):
        self.max_duration = max_duration
        self.sample_rate = 16000
        self.buffer = np.array([], dtype=np.int16)
        self.temp_dir = tempfile.mkdtemp()
        self.last_timestamp = None
        self.total_samples = 0
        self.complete_buffer = np.array([], dtype=np.int16)  # Store all audio
        
    def add_audio(self, audio_data: np.ndarray, timestamp: float = None):
        try:
            if len(audio_data) == 0:
                return

            # Ensure audio is the right type
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            
            # Ensure audio is 1D
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()

            # Add to rolling buffer
            self.buffer = np.append(self.buffer, audio_data)
            self.total_samples += len(audio_data)
            self.last_timestamp = timestamp

            # Also add to complete buffer
            self.complete_buffer = np.append(self.complete_buffer, audio_data)

            # Trim rolling buffer if it exceeds max duration
            max_samples = int(self.sample_rate * self.max_duration)
            if len(self.buffer) > max_samples:
                excess = len(self.buffer) - max_samples
                self.buffer = self.buffer[excess:]
                self.total_samples = len(self.buffer)

            # Calculate actual duration
            duration = self.get_duration()
            
            # Only log if we have meaningful audio data
            if np.any(audio_data != 0):
                logger.info(f"Added audio data: {len(audio_data)} samples, "
                          f"Buffer duration: {duration:.2f}s, "
                          f"Total samples: {self.total_samples}, "
                          f"Range: [{audio_data.min()}, {audio_data.max()}]")

        except Exception as e:
            logger.error(f"Error adding audio to buffer: {e}")

    def get_complete_audio(self) -> np.ndarray:
        """Get the complete audio buffer."""
        return self.complete_buffer.copy()

    def get_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return len(self.buffer) / self.sample_rate

    def get_complete_duration(self) -> float:
        """Get total duration of all recorded audio in seconds."""
        return len(self.complete_buffer) / self.sample_rate

    def get_audio_file(self) -> Optional[str]:
        """Get current buffer as WAV file."""
        if len(self.buffer) == 0:
            return None
        
        try:    
            temp_path = os.path.join(self.temp_dir, f"audio_{datetime.now().timestamp()}.wav")
            duration = self.get_duration()
            
            # Write buffer to WAV file
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(self.buffer.tobytes())
                
            logger.debug(f"Saved audio file: {temp_path}, "
                        f"Duration: {duration:.2f}s, "
                        f"Samples: {len(self.buffer)}")
                
            return temp_path
                
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return None

    def clear(self):
        """Clear the rolling buffer but keep complete buffer."""
        self.buffer = np.array([], dtype=np.int16)
        self.total_samples = 0
        self.last_timestamp = None

    def cleanup(self):
        """Clean up temporary files."""
        try:
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up audio buffer: {e}")