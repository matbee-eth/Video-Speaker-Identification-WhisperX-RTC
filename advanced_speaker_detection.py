import cv2
import numpy as np
import webrtcvad
from typing import List, Tuple, Optional, Dict
import mediapipe as mp
from dataclasses import dataclass
import numpy.typing as npt
import base64
import logging
from collections import deque
from scipy.spatial.distance import euclidean

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class FaceData:
    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray
    is_speaking: bool
    confidence: float
    lip_movement_score: float

class AdvancedSpeakerDetector:
    def __init__(
        self,
        vad_aggressiveness: int = 3,
        temporal_window: int = 10,
        min_speech_frames: int = 3,
        sample_rate: int = 16000,
        frame_duration: int = 30,  # in milliseconds
        lip_movement_threshold: float = 0.15,
        correlation_window: int = 5
    ):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.temporal_window = temporal_window
        self.min_speech_frames = min_speech_frames
        self.lip_movement_threshold = lip_movement_threshold
        self.correlation_window = correlation_window
        
        # History tracking for each face
        self.speaking_histories: Dict[int, deque] = {}
        self.lip_movement_histories: Dict[int, deque] = {}
        self.previous_faces: Dict[int, FaceData] = {}  # Changed from previous_landmarks
        
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)

    def calculate_lip_movement(
        self,
        current_landmarks: np.ndarray,
        previous_landmarks: Optional[np.ndarray]
    ) -> float:
        """Calculate lip movement using MediaPipe Face Detection landmarks."""
        if previous_landmarks is None or len(current_landmarks) < 2:
            return 0.0
            
        try:
            # Calculate mouth metrics using available landmarks
            current_mouth_width = euclidean(current_landmarks[3], current_landmarks[4])
            current_mouth_height = euclidean(current_landmarks[2], 
                                           (current_landmarks[3] + current_landmarks[4]) / 2)
            
            if previous_landmarks is not None:
                prev_mouth_width = euclidean(previous_landmarks[3], previous_landmarks[4])
                prev_mouth_height = euclidean(previous_landmarks[2], 
                                            (previous_landmarks[3] + previous_landmarks[4]) / 2)
                
                # Calculate relative changes
                width_change = abs(current_mouth_width - prev_mouth_width) / prev_mouth_width
                height_change = abs(current_mouth_height - prev_mouth_height) / prev_mouth_height
                
                # Combine changes into a movement score
                movement_score = (width_change + height_change) / 2
                return movement_score
            
        except Exception as e:
            logger.error(f"Error calculating lip movement: {e}")
            
        return 0.0

    def track_face_identity(
        self,
        current_faces: List[FaceData],
        previous_faces: Dict[int, FaceData]
    ) -> Dict[int, int]:
        """Track face identities across frames using IOU of bounding boxes."""
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
            
            # Calculate union
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
            if iou_matrix[i, j] > 0.3:  # Lowered IOU threshold for better tracking
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

    def extract_face_data(self, frame: np.ndarray) -> List[FaceData]:
        """Extract face data with detections and landmarks."""
        if frame is None:
            logger.error("Received empty frame")
            return []

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]

        # Process the frame
        results = self.face_detection.process(rgb_frame)
        faces = []

        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Ensure coordinates are within frame bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = min(w, width - x)
                h = min(h, height - y)

                # Create landmark array
                landmarks = np.array([
                    [kp.x * width, kp.y * height]
                    for kp in detection.location_data.relative_keypoints
                ])

                face_data = FaceData(
                    bbox=(x, y, w, h),
                    landmarks=landmarks,
                    is_speaking=False,
                    confidence=detection.score[0],
                    lip_movement_score=0.0
                )
                faces.append(face_data)

        return faces

    def process_frame(
        self,
        frame_data: str,
        audio_data: str
    ) -> List[dict]:
        """Process a frame with synchronized audio-visual analysis."""
        try:
            # Decode frame
            frame = self.decode_frame(frame_data)
            if frame is None:
                return []
                
            # Get face detections
            faces = self.extract_face_data(frame)
            
            # Track face identities
            identity_map = self.track_face_identity(faces, self.previous_faces)
            
            # Process audio
            is_speech = self.process_audio(audio_data)
            
            # Update speaking status and lip movement for each face
            for i, face in enumerate(faces):
                face_id = identity_map[i]
                
                # Calculate lip movement using the previous face's landmarks
                prev_landmarks = (self.previous_faces[face_id].landmarks 
                                if face_id in self.previous_faces 
                                else None)
                movement_score = self.calculate_lip_movement(
                    face.landmarks,
                    prev_landmarks
                )
                face.lip_movement_score = movement_score
                
                # Initialize histories if needed
                if face_id not in self.speaking_histories:
                    self.speaking_histories[face_id] = deque(maxlen=self.temporal_window)
                    self.lip_movement_histories[face_id] = deque(maxlen=self.correlation_window)
                
                # Update histories
                self.speaking_histories[face_id].append(is_speech)
                self.lip_movement_histories[face_id].append(movement_score)
                
                # Determine speaking status based on both VAD and lip movement
                speaking_frames = sum(self.speaking_histories[face_id])
                avg_movement = np.mean(self.lip_movement_histories[face_id])
                
                # A face is considered speaking if:
                # 1. There is voice activity detected
                # 2. The face shows significant lip movement
                # 3. The face has consistent speech frames in its history
                face.is_speaking = (
                    speaking_frames >= self.min_speech_frames and
                    avg_movement >= self.lip_movement_threshold and
                    is_speech  # Current frame must have speech
                )
            
            # Update previous faces
            self.previous_faces = {
                identity_map[i]: face
                for i, face in enumerate(faces)
            }
            
            # Convert to serializable format
            results = [
                {
                    "bbox": list(face.bbox),
                    "landmarks": face.landmarks.tolist(),
                    "is_speaking": bool(face.is_speaking),  # Convert np.bool_ to Python bool
                    "confidence": float(face.confidence),
                    "lip_movement_score": float(face.lip_movement_score)
                }
                for face in faces
            ]
            
            return results
            
        except Exception as e:
            logger.error(f"Error in process_frame: {e}", exc_info=True)
            return []

    def decode_frame(self, frame_data: str) -> Optional[np.ndarray]:
        """Decode base64 frame data."""
        try:
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return None
            
    def process_audio(self, audio_data: str) -> bool:
        """Process audio data for voice activity detection."""
        try:
            if isinstance(audio_data, str):
                audio_data = base64.b64decode(audio_data)
            chunk_size = self.frame_size * 2
            chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
            return any(self.vad.is_speech(chunk, self.sample_rate) for chunk in chunks if len(chunk) == chunk_size)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return False

    def __del__(self):
        """Cleanup resources."""
        self.face_detection.close()