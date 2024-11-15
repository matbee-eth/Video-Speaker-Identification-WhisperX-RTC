import whisperx
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging
from advanced_speaker_detection import AdvancedSpeakerDetector, FaceData
import wave
from datetime import datetime
from scipy import signal

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    text: Optional[str] = None

@dataclass
class EnhancedFaceData(FaceData):
    speaker_id: Optional[str] = None
    confidence_score: float = 0.0

class AudioVisualSpeakerDetector:
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "large-v2",
        compute_type: str = "float16",
        batch_size: int = 16,
        hf_token: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ):
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # Initialize base speaker detector
        self.visual_detector = AdvancedSpeakerDetector()
        
        # Initialize WhisperX components
        self.whisper_model = whisperx.load_model(model_name, device, compute_type=compute_type)
        self.diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=device
        )
        
        # Speaker tracking
        self.face_speaker_mapping: Dict[int, str] = {}  # face_id -> speaker_id
        self.speaker_history: Dict[str, List[Tuple[float, int]]] = defaultdict(list)  # speaker_id -> [(timestamp, face_id)]
        self.confidence_threshold = 0.7
        self.debug_audio = True
    
    def _save_debug_audio(self, audio: np.ndarray, sample_rate: int = 48000) -> None:
        """Save audio array as WAV file for debugging purposes."""
        if not self.debug_audio:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_filename = f"debug_audio_{timestamp}.wav"
            
            logger.info(f"Saving debug audio to {debug_filename}")
            
            # Ensure audio is in int16 format
            if audio.dtype != np.int16:
                if audio.dtype in [np.float32, np.float64]:
                    if np.abs(audio).max() <= 1.0:
                        audio = (audio * 32767).astype(np.int16)
                    else:
                        audio = audio.astype(np.int16)
                else:
                    audio = audio.astype(np.int16)
            
            with wave.open(debug_filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(48000)  # Use the actual sample rate
                wav_file.writeframes(audio.tobytes())
                
            logger.info(f"Saved debug audio: {debug_filename}, "
                    f"samples={len(audio)}, rate={sample_rate}Hz, "
                    f"duration={len(audio)/sample_rate:.2f}s")
            
        except Exception as e:
            logger.error(f"Error saving debug audio: {e}", exc_info=True)

    def process_video_segment(
        self,
        audio: np.ndarray,
        video_frames: List[Tuple[float, np.ndarray]]
    ) -> Tuple[List[Dict], List[Tuple[float, List[Dict]]]]:
        try:
            logger.info(f"Processing audio data: shape={audio.shape}, dtype={audio.dtype}")
            
            # Ensure audio is float32 and normalized for whisperx
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            
            result = self.whisper_model.transcribe(audio, batch_size=self.batch_size)
            
            # Align whisper output
            align_model, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=self.device
            )
            
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            
            # Run speaker diarization
            diarize_segments = self.diarize_model(
                audio,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )
            
            # Process video frames
            face_detections = []
            for timestamp, frame in video_frames:
                # Get face detections
                current_faces = self._detect_faces(frame)
                
                # Find active speakers
                active_speakers = self._get_active_speakers(result["segments"], timestamp)
                
                # Update speaker mappings
                enhanced_faces = self._update_speaker_mappings(
                    timestamp,
                    current_faces,
                    active_speakers
                )
                
                face_detections.append((timestamp, enhanced_faces))
            
            return result["segments"], face_detections
            
        except Exception as e:
            logger.error(f"Error in process_video_segment: {e}", exc_info=True)
            return [], []
        finally:
            torch.cuda.empty_cache()
            
    def _get_active_speakers(
        self,
        segments: List[Dict],
        timestamp: float
    ) -> List[Dict]:
        """Find speaker segments active at the given timestamp."""
        return [
            segment for segment in segments
            if segment["start"] <= timestamp <= segment["end"]
        ]
        
    def _update_speaker_mappings(
        self,
        timestamp: float,
        faces: List[FaceData],
        active_speakers: List[Dict]
    ):
        """Update face-to-speaker mappings based on visual and audio cues."""
        # Find speaking faces
        speaking_faces = [
            (i, face) for i, face in enumerate(faces)
            if face.is_speaking
        ]
        
        # Update mappings based on temporal correlation
        for speaker_segment in active_speakers:
            speaker_id = speaker_segment["speaker"]
            
            # Find best matching face based on speaking activity and consistency
            best_face_id = None
            best_confidence = 0
            
            for face_idx, face in speaking_faces:
                # Calculate confidence based on:
                # 1. Lip movement correlation
                # 2. Temporal consistency with previous assignments
                # 3. Face detection confidence
                confidence = self._calculate_mapping_confidence(
                    face, speaker_id, timestamp, face_idx
                )
                
                if confidence > best_confidence and confidence > self.confidence_threshold:
                    best_confidence = confidence
                    best_face_id = face_idx
            
            # Update mapping if confident match found
            if best_face_id is not None:
                self.face_speaker_mapping[best_face_id] = speaker_id
                self.speaker_history[speaker_id].append((timestamp, best_face_id))
                
                # Prune old history entries
                self._prune_speaker_history(speaker_id)
                
    def _calculate_mapping_confidence(
        self,
        face: FaceData,
        speaker_id: str,
        timestamp: float,
        face_id: int
    ) -> float:
        """Calculate confidence score for face-speaker mapping."""
        confidence = 0.0
        
        # Weight factors
        MOVEMENT_WEIGHT = 0.4
        HISTORY_WEIGHT = 0.4
        DETECTION_WEIGHT = 0.2
        
        # 1. Lip movement score
        confidence += face.lip_movement_score * MOVEMENT_WEIGHT
        
        # 2. Historical consistency
        if speaker_id in self.speaker_history:
            recent_history = [
                entry for entry in self.speaker_history[speaker_id]
                if abs(entry[0] - timestamp) < 5.0  # Look at last 5 seconds
            ]
            if recent_history:
                history_score = sum(
                    1 for _, hist_face_id in recent_history
                    if hist_face_id == face_id
                ) / len(recent_history)
                confidence += history_score * HISTORY_WEIGHT
                
        # 3. Face detection confidence
        confidence += face.confidence * DETECTION_WEIGHT
        
        return confidence
        
    def _prune_speaker_history(self, speaker_id: str, max_history: float = 10.0):
        """Remove history entries older than max_history seconds."""
        if speaker_id in self.speaker_history:
            latest_timestamp = max(t for t, _ in self.speaker_history[speaker_id])
            self.speaker_history[speaker_id] = [
                entry for entry in self.speaker_history[speaker_id]
                if latest_timestamp - entry[0] <= max_history
            ]
            
    def _enhance_face_detections(
        self,
        faces: List[FaceData],
        timestamp: float
    ) -> List[Dict]:
        """Enhance face detections with speaker information."""
        enhanced_faces = []
        
        for i, face in enumerate(faces):
            speaker_id = self.face_speaker_mapping.get(i)
            confidence = self._calculate_mapping_confidence(
                face, speaker_id, timestamp, i
            ) if speaker_id else 0.0
            
            enhanced_faces.append({
                "bbox": list(face.bbox),
                "landmarks": face.landmarks.tolist(),
                "is_speaking": bool(face.is_speaking),
                "confidence": float(face.confidence),
                "lip_movement_score": float(face.lip_movement_score),
                "speaker_id": speaker_id,
                "speaker_confidence": float(confidence)
            })
            
        return enhanced_faces