import logging
import wave
import tempfile
import os
import cv2
import numpy as np
import whisperx
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import torch
from scipy.spatial.distance import euclidean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FaceData:
    bbox: Tuple[float, float, float, float]
    landmarks: np.ndarray
    is_speaking: bool
    confidence: float
    lip_movement_score: float
    speaker_id: Optional[int] = None
    speaker_confidence: float = 0.0

class EnhancedSpeakerDetector:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_name: str = "large-v2",
        compute_type: str = "float16",
        batch_size: int = 16,
        hf_token: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        temporal_window: int = 10,
        lip_movement_threshold: float = 0.15
    ):
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.temporal_window = temporal_window
        self.lip_movement_threshold = lip_movement_threshold
        
        # Initialize WhisperX components
        logger.info("Loading WhisperX model...")
        self.whisper_model = whisperx.load_model(model_name, device, compute_type=compute_type)
        
        logger.info("Loading diarization pipeline...")
        self.diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=device
        )
        
        # Initialize state tracking
        self.speaker_histories: Dict[int, deque] = {}
        self.lip_movement_histories: Dict[int, deque] = {}
        self.previous_faces: Dict[int, FaceData] = {}
        self.face_speaker_mapping: Dict[int, int] = {}
        
        # Configuration
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.confidence_threshold = 0.7
        
        # Create temp directory for audio processing
        self.temp_dir = tempfile.mkdtemp()
        
    def calculate_lip_movement(
        self,
        current_landmarks: np.ndarray,
        previous_landmarks: Optional[np.ndarray]
    ) -> float:
        """Calculate lip movement score using facial landmarks."""
        if previous_landmarks is None or len(current_landmarks) < 2:
            return 0.0
            
        try:
            # Calculate mouth metrics
            current_mouth_width = euclidean(current_landmarks[3], current_landmarks[4])
            current_mouth_height = euclidean(
                current_landmarks[2],
                (current_landmarks[3] + current_landmarks[4]) / 2
            )
            
            prev_mouth_width = euclidean(previous_landmarks[3], previous_landmarks[4])
            prev_mouth_height = euclidean(
                previous_landmarks[2],
                (previous_landmarks[3] + previous_landmarks[4]) / 2
            )
            
            # Calculate relative changes
            width_change = abs(current_mouth_width - prev_mouth_width) / prev_mouth_width
            height_change = abs(current_mouth_height - prev_mouth_height) / prev_mouth_height
            
            # Combine into movement score
            return (width_change + height_change) / 2
            
        except Exception as e:
            logger.error(f"Error calculating lip movement: {e}")
            return 0.0

    def track_face_identity(
        self,
        current_faces: List[FaceData],
        previous_faces: Dict[int, FaceData]
    ) -> Dict[int, int]:
        """Track face identities across frames using IOU matching."""
        if not previous_faces:
            return {i: i for i in range(len(current_faces))}
            
        def calculate_iou(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # Calculate intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
                
            intersection = (x_right - x_left) * (y_bottom - y_top)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        # Calculate IOU matrix
        iou_matrix = np.zeros((len(current_faces), len(previous_faces)))
        for i, curr_face in enumerate(current_faces):
            for j, prev_face in previous_faces.items():
                iou = calculate_iou(curr_face.bbox, prev_face.bbox)
                iou_matrix[i, j] = iou
                
        # Assign identities based on maximum IOU
        identity_map = {}
        while len(identity_map) < len(current_faces):
            if iou_matrix.size == 0:
                break
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            if iou_matrix[i, j] > 0.3:
                identity_map[i] = j
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1
            
        # Assign new IDs to unmatched faces
        max_id = max(previous_faces.keys()) + 1 if previous_faces else 0
        for i in range(len(current_faces)):
            if i not in identity_map:
                identity_map[i] = max_id
                max_id += 1
                
        return identity_map

    def process_video_segment(
        self,
        audio_path: str,
        video_frames: List[Tuple[float, np.ndarray]]
    ) -> Tuple[List[Dict], List[Tuple[float, List[Dict]]]]:
        try:
            logger.info(f"Processing audio file: {audio_path}")
            
            # Load and process audio with WhisperX
            audio = whisperx.load_audio(audio_path)
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
            
            # Combine results
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Process video frames
            face_detections = []
            for timestamp, frame in video_frames:
                # Get face detections
                current_faces = self._detect_faces(frame)
                
                # Find active speakers at current timestamp
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
            # Cleanup CUDA memory
            torch.cuda.empty_cache()

    def _detect_faces(self, frame: np.ndarray) -> List[FaceData]:
        """Detect faces and extract landmarks using MediaPipe."""
        # This should use your existing face detection implementation
        # Placeholder for now
        return []

    def _get_active_speakers(self, segments: List[Dict], timestamp: float) -> List[Dict]:
        """Get speaker segments active at the given timestamp."""
        return [
            segment for segment in segments
            if segment["start"] <= timestamp <= segment["end"]
        ]

    def _update_speaker_mappings(
        self,
        timestamp: float,
        faces: List[FaceData],
        active_speakers: List[Dict]
    ) -> List[Dict]:
        """Update face-speaker mappings and enhance face detections."""
        # Track face identities
        identity_map = self.track_face_identity(faces, self.previous_faces)
        
        # Update histories and calculate lip movement
        enhanced_faces = []
        for i, face in enumerate(faces):
            face_id = identity_map[i]
            
            # Initialize histories if needed
            if face_id not in self.speaker_histories:
                self.speaker_histories[face_id] = deque(maxlen=self.temporal_window)
                self.lip_movement_histories[face_id] = deque(maxlen=self.temporal_window)
            
            # Calculate lip movement
            prev_face = self.previous_faces.get(face_id)
            movement_score = self.calculate_lip_movement(
                face.landmarks,
                prev_face.landmarks if prev_face else None
            )
            
            # Update histories
            self.lip_movement_histories[face_id].append(movement_score)
            
            # Match with active speakers
            matched_speaker = None
            max_confidence = 0
            
            for speaker in active_speakers:
                confidence = self._calculate_speaker_confidence(
                    face, speaker, timestamp, movement_score
                )
                if confidence > max_confidence and confidence > self.confidence_threshold:
                    max_confidence = confidence
                    matched_speaker = speaker["speaker"]
            
            # Create enhanced face detection
            enhanced_face = {
                "bbox": list(face.bbox),
                "landmarks": face.landmarks.tolist(),
                "is_speaking": matched_speaker is not None,
                "confidence": float(face.confidence),
                "lip_movement_score": float(movement_score),
                "speaker_id": matched_speaker,
                "speaker_confidence": float(max_confidence)
            }
            
            enhanced_faces.append(enhanced_face)
            
        # Update previous faces
        self.previous_faces = {
            identity_map[i]: face
            for i, face in enumerate(faces)
        }
        
        return enhanced_faces

    def _calculate_speaker_confidence(
        self,
        face: FaceData,
        speaker: Dict,
        timestamp: float,
        movement_score: float
    ) -> float:
        """Calculate confidence score for speaker-face matching."""
        # Weights for different factors
        MOVEMENT_WEIGHT = 0.4
        TEMPORAL_WEIGHT = 0.4
        DETECTION_WEIGHT = 0.2
        
        confidence = 0.0
        
        # Lip movement score
        avg_movement = np.mean(list(self.lip_movement_histories.get(id(face), [])) + [movement_score])
        confidence += avg_movement * MOVEMENT_WEIGHT
        
        # Temporal consistency
        temporal_score = 0.0
        if speaker["speaker"] in self.face_speaker_mapping:
            last_face_id = self.face_speaker_mapping[speaker["speaker"]]
            if last_face_id == id(face):
                temporal_score = 1.0
        confidence += temporal_score * TEMPORAL_WEIGHT
        
        # Face detection confidence
        confidence += face.confidence * DETECTION_WEIGHT
        
        return confidence

    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Remove temporary files
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")