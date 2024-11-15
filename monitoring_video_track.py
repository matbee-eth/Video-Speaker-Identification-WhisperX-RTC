
from datetime import datetime
from aiortc import (
    VideoStreamTrack
)
import av
import cv2
class MonitoringVideoTrack(VideoStreamTrack):
    def __init__(self, track, width=1280, height=720):
        super().__init__()
        self.track = track
        self.width = width
        self.height = height
        self.frame_count = 0
        
    async def recv(self):
        frame = await self.track.recv()
        
        # Process the frame
        img = frame.to_ndarray(format="bgr24")
        
        # Resize if needed
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height))
            
        # Add frame counter and timestamp
        self.frame_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(
            img,
            f"Frame: {self.frame_count} | Time: {timestamp}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        # Convert back to VideoFrame
        new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        
        return new_frame