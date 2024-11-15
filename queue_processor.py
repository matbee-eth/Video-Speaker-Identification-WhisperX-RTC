import asyncio
import json
import os
from queue import Queue
import threading
import time
from typing import List, Optional
import logging

from audio_visual_speaker_detection import AudioVisualSpeakerDetector
from real_time_audio_buffer import RealtimeAudioBuffer
from media_frame import MediaFrame
from aiortc import RTCDataChannel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueueProcessor:
    def __init__(self, batch_duration: float = 5.0, speaker_detector = None, min_audio_duration: float = 3.0):
        self.batch_duration = batch_duration
        self.min_audio_duration = min_audio_duration
        self.frame_queue = Queue()
        self.is_processing = False
        self.current_batch: List[MediaFrame] = []
        self.last_process_time = 0
        self.processing_lock = threading.Lock()
        self.audio_buffer = RealtimeAudioBuffer(max_duration=120)
        self.speaker_detector = speaker_detector
        self.data_channel = None
        self.loop = asyncio.new_event_loop()
        self._running = True
        self._last_log_time = 0  # Add timestamp for last log
        self.LOG_INTERVAL = 5.0  # Log every 5 seconds
        self._start_processing_thread()

    def _start_processing_thread(self):
        def process_loop():
            asyncio.set_event_loop(self.loop)
            while self._running:
                try:
                    current_time = time.time()
                    time_since_last = current_time - self.last_process_time
                    buffer_duration = self.audio_buffer.get_duration()
                    has_frames = not self.frame_queue.empty()
                    
                    # Only log periodically when we have meaningful content
                    should_log = (
                        has_frames and 
                        buffer_duration > 0 and 
                        current_time - self._last_log_time >= self.LOG_INTERVAL
                    )
                    
                    if should_log:
                        logger.info(
                            f"Processing conditions: "
                            f"Time since last={time_since_last:.2f}s, "
                            f"Buffer duration={buffer_duration:.2f}s, "
                            f"Has frames={has_frames}, "
                            f"Is processing={self.is_processing}"
                        )
                        self._last_log_time = current_time
                    
                    # Check if we should process
                    should_process = (
                        time_since_last >= self.batch_duration and
                        buffer_duration >= self.min_audio_duration and
                        has_frames and
                        not self.is_processing
                    )
                    
                    if should_process:
                        logger.info(
                            f"Starting batch processing with "
                            f"buffer duration={buffer_duration:.2f}s"
                        )
                        self.loop.run_until_complete(self._process_batch())
                        
                    time.sleep(0.1)  # Prevent tight loop
                    
                except RuntimeError as e:
                    if "Event loop is closed" in str(e):
                        break
                    else:
                        logger.error(f"Error in processing loop: {e}")
                except Exception as e:
                    logger.error(f"Error in processing loop: {e}")

        self.processing_thread = threading.Thread(target=process_loop, daemon=True)
        self.processing_thread.start()
        
    def add_frame(self, frame: MediaFrame):
        """Add a frame to the processing queue and update audio duration."""
        try:
            # Add frame to queue
            self.frame_queue.put(frame)
            
            # Add audio data to buffer if present
            if frame.audio_data is not None and len(frame.audio_data) > 0:
                self.audio_buffer.add_audio(frame.audio_data, frame.timestamp)
                
                # Log queue and buffer status
                buffer_duration = self.audio_buffer.get_duration()
                queue_size = self.frame_queue.qsize()
                
                logger.info(
                    f"Added frame: Queue size={queue_size}, "
                    f"Buffer duration={buffer_duration:.2f}s"
                )
                
        except Exception as e:
            logger.error(f"Error adding frame: {e}")

    async def _process_batch(self):
        """Process a batch of frames with speaker detection."""
        with self.processing_lock:
            if self.is_processing:
                return
            self.is_processing = True
            
        try:
            # Get current buffer duration
            buffer_duration = self.audio_buffer.get_duration()
            logger.info(f"Processing batch with buffer duration: {buffer_duration:.2f}s")
            
            if buffer_duration < self.min_audio_duration:
                logger.debug(f"Insufficient audio ({buffer_duration:.2f}s < {self.min_audio_duration}s)")
                return
                
            # Collect frames from queue
            batch_frames = []
            while not self.frame_queue.empty():
                batch_frames.append(self.frame_queue.get())
                
            if not batch_frames:
                logger.debug("No frames to process")
                return

            # Sort frames by timestamp
            batch_frames.sort(key=lambda x: x.timestamp)
            logger.info(f"Processing {len(batch_frames)} frames")

            # Get audio file
            audio_path = self.audio_buffer.get_audio_file()
            if not audio_path:
                logger.warning("Failed to get audio file")
                return

            try:
                # Process video frames with speaker detection
                video_segments = [
                    (frame.timestamp, frame.video_frame)
                    for frame in batch_frames
                    if frame.video_frame is not None
                ]

                if video_segments and self.speaker_detector:
                    results = self.speaker_detector.process_video_segment(
                        audio_path,
                        video_segments
                    )
                    
                    # Send results if we have a data channel
                    if self.data_channel and self.data_channel.readyState == "open":
                        for i, (timestamp, _) in enumerate(video_segments):
                            if results[1] and i < len(results[1]):
                                message = {
                                    "type": "detection_results",
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
                                    "face_detections": results[1][i][1] if results[1][i][1] else []
                                }
                                
                                await self.loop.run_in_executor(
                                    None,
                                    lambda: self.data_channel.send(json.dumps(message))
                                )
                                
                    logger.info(f"Processed batch with {len(video_segments)} segments")

            finally:
                # Cleanup audio file
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.error(f"Error removing audio file: {e}")
                
            # Update last process time
            self.last_process_time = time.time()
            
            # Clear the audio buffer
            self.audio_buffer.clear()

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
        finally:
            self.is_processing = False

    def cleanup(self):
        """Clean up resources."""
        self._running = False
        self.audio_buffer.cleanup()
        if not self.loop.is_closed():
            self.loop.stop()
            # Don't close the loop here, as it might be in use