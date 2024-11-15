
from aiortc import (
    VideoStreamTrack
)
import av
import cv2
import numpy as np

from media_frame import MediaFrame
class ProcessedVideoTrack(VideoStreamTrack):
    def __init__(self, track, processor):
        super().__init__()
        self.track = track
        self.processor = processor
        self.prev_pts = None
        self.latest_detections = []
        
    async def recv(self):
        frame = await self.track.recv()
        
        if self.prev_pts is None:
            self.prev_pts = frame.pts
        else:
            frame.pts = max(self.prev_pts + 1, frame.pts)
            self.prev_pts = frame.pts
            
        # Convert frame to numpy array for processing
        image = frame.to_ndarray(format="bgr24")
        timestamp = frame.time
        
        # Add frame to processing queue
        self.processor.queue_processor.add_frame(MediaFrame(
            timestamp=timestamp,
            video_frame=image.copy()
        ))
        
        # Draw latest detections
        self.draw_detections(image)
        
        # Convert back to VideoFrame
        new_frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        
        return new_frame
        
    def draw_detections(self, image):
        for detection in self.latest_detections:
            bbox = detection["bbox"]
            is_speaking = detection["is_speaking"]
            speaker_id = detection.get("speaker_id")
            confidence = detection.get("speaker_confidence", 0)
            
            # Draw bounding box
            color = (0, 255, 0) if is_speaking else (0, 0, 255)
            if speaker_id is not None:
                hue = (speaker_id * 40) % 180
                color = tuple(int(x) for x in cv2.cvtColor(
                    np.uint8([[[hue, 255, 255]]]), 
                    cv2.COLOR_HSV2BGR)[0][0]
                )
            
            cv2.rectangle(
                image, 
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                color, 2
            )
            
            # Draw labels
            label = f"Speaker {speaker_id}" if speaker_id is not None else (
                "Speaking" if is_speaking else "Not Speaking"
            )
            conf_text = f"Conf: {confidence:.1%}" if speaker_id is not None else ""
            
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1] - 45)),
                (int(bbox[0] + bbox[2]), int(bbox[1])),
                (0, 0, 0, 128),
                -1
            )
            
            cv2.putText(
                image, label,
                (int(bbox[0] + 5), int(bbox[1] - 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
            if conf_text:
                cv2.putText(
                    image, conf_text,
                    (int(bbox[0] + 5), int(bbox[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
