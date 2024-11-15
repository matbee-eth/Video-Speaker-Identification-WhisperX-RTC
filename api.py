from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import tempfile
import os
from typing import Optional, Dict
import base64
import numpy as np
from pydantic import BaseModel

from audio_visual_speaker_detection import AudioVisualSpeakerDetector

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessingConfig(BaseModel):
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    hf_token: Optional[str] = None

class DetectionState:
    def __init__(self):
        self.detector: Optional[AudioVisualSpeakerDetector] = None
        self.audio_buffer: Dict[str, bytes] = {}
        self.latest_audio_timestamp: Dict[str, float] = {}
        
    def initialize_detector(self, config: ProcessingConfig):
        if self.detector is None:
            self.detector = AudioVisualSpeakerDetector(
                hf_token=config.hf_token,
                min_speakers=config.min_speakers,
                max_speakers=config.max_speakers
            )

state = DetectionState()

@app.post("/initialize")
async def initialize_detection(config: ProcessingConfig):
    try:
        state.initialize_detector(config)
        return {"status": "success", "message": "Detector initialized"}
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        return {"status": "error", "message": str(e)}
import wave
import struct

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    detector = AudioVisualSpeakerDetector()
    session_id = os.urandom(16).hex()
    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    audio_file.close()  # Close it so wave can open it
    
    # Initialize WAV file
    wav_file = None
    accumulated_audio = bytearray()
    
    try:
        while True:
            data = await websocket.receive_json()
            timestamp = data.get("timestamp", 0.0)
            
            if "audio" in data:
                try:
                    # Decode base64 to bytes
                    audio_bytes = base64.b64decode(data["audio"])
                    
                    # Convert bytes to Int16Array
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    # Accumulate audio data
                    accumulated_audio.extend(audio_data.tobytes())
                    
                    # Write complete WAV file
                    with wave.open(audio_file.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # mono
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(16000)  # 16kHz
                        wav_file.writeframes(accumulated_audio)
                    
                except Exception as e:
                    logger.error(f"Error processing audio data: {e}")
                    continue
            
            # Process frame
            if "frame" in data:
                frame = detector.visual_detector.decode_frame(data["frame"])
                if frame is not None:
                    try:
                        # Process frame with audio context
                        results = detector.process_video_segment(
                            audio_file.name,
                            [(timestamp, frame)]
                        )
                        
                        # Send results
                        await websocket.send_json({
                            "timestamp": timestamp,
                            "speaker_segments": [
                                {
                                    "start": segment.start,
                                    "end": segment.end,
                                    "speaker": segment.speaker,
                                    "text": segment.text
                                }
                                for segment in results[0]
                            ],
                            "face_detections": results[1][0][1] if results[1] else []
                        })
                    except Exception as e:
                        logger.error(f"Error processing video frame: {e}")
                        continue
                    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "status": "error",
            "message": str(e)
        })
    finally:
        # Cleanup
        logger.info("WebSocket connection closed")
        try:
            os.unlink(audio_file.name)
        except:
            pass
        if session_id in state.audio_buffer:
            del state.audio_buffer[session_id]
        if session_id in state.latest_audio_timestamp:
            del state.latest_audio_timestamp[session_id]
        await websocket.close()
        
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)