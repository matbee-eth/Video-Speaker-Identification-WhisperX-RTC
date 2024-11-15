import asyncio
from datetime import datetime
import os
import tempfile
from typing import Dict, List, Optional, Set, Union
from aiortc.contrib.media import MediaRecorder, MediaRelay

from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCDataChannel,
)
import av
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from monitoring_video_track import MonitoringVideoTrack
from processed_audio_track import ProcessedAudioTrack
from processed_video_track import ProcessedVideoTrack
from queue_processor import QueueProcessor
class MediaStreamProcessor:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.audio_relay = MediaRelay()
        self.video_relay = MediaRelay()
        self.active_tracks: Set[MediaStreamTrack] = set()
        self.temp_dir = tempfile.mkdtemp()
        self.current_recorder: Optional[MediaRecorder] = None
        self.queue_processor = QueueProcessor(batch_duration=5.0)
        self.original_tracks: Dict[str, MediaStreamTrack] = {}
        self.processed_tracks: Dict[str, Union[ProcessedVideoTrack, ProcessedAudioTrack]] = {}

    def create_audio_transformer(self):
        """Create an audio transformer to ensure mono output"""
        transformer = av.AudioResampler(
            format=av.AudioFormat('s16'),
            layout='mono',
            rate=16000
        )
        return transformer

    async def add_track(self, track: MediaStreamTrack, pc: RTCPeerConnection) -> MediaStreamTrack:
        """Process and relay a new media track."""
        self.original_tracks[track.kind] = track
        
        if track.kind == "video":
            processed_track = ProcessedVideoTrack(track, self)
            self.processed_tracks["video"] = processed_track
            relayed = self.video_relay.subscribe(processed_track)
        else:  # audio
            # Create processed track
            processed_track = ProcessedAudioTrack(track, self)
            self.processed_tracks["audio"] = processed_track
            
            await processed_track.start()  # Start the audio processing
            relayed = self.audio_relay.subscribe(processed_track)
            
        self.active_tracks.add(relayed)
        return relayed

    async def cleanup(self):
        """Clean up resources."""
        try:
            # Stop recording first
            await self.stop_recording()
            await asyncio.sleep(0.5)  # Wait for recording to finish
            
            # Stop all tracks
            for track in list(self.active_tracks):
                if track and hasattr(track, 'stop'):
                    if asyncio.iscoroutinefunction(track.stop):
                        await track.stop()
                    else:
                        track.stop()
            self.active_tracks.clear()
            
            # Stop processed tracks
            for track in self.processed_tracks.values():
                if track and hasattr(track, 'stop'):
                    if asyncio.iscoroutinefunction(track.stop):
                        await track.stop()
                    else:
                        track.stop()
            self.processed_tracks.clear()
            
            # Stop original tracks
            for track in self.original_tracks.values():
                if track and hasattr(track, 'stop'):
                    if asyncio.iscoroutinefunction(track.stop):
                        await track.stop()
                    else:
                        track.stop()
            self.original_tracks.clear()
            
            # Cleanup queue processor
            if self.queue_processor:
                self.queue_processor.cleanup()
            
            # Clean up temp directory
            try:
                # Wait a moment for any pending file operations
                await asyncio.sleep(0.5)
                
                if os.path.exists(self.temp_dir):
                    for file in os.listdir(self.temp_dir):
                        file_path = os.path.join(self.temp_dir, file)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            logger.warning(f"Error removing file {file_path}: {e}")
                    try:
                        os.rmdir(self.temp_dir)
                    except Exception as e:
                        logger.warning(f"Error removing temp directory: {e}")
            except Exception as e:
                logger.error(f"Error cleaning up temp directory: {e}")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)


    def set_data_channel(self, data_channel: RTCDataChannel):
        """Set the data channel for sending detection results."""
        self.queue_processor.data_channel = data_channel

    async def create_monitoring_tracks(self) -> List[MediaStreamTrack]:
        """Create monitoring tracks from original tracks."""
        monitoring_tracks = []
        
        if "video" in self.original_tracks:
            video_track = MonitoringVideoTrack(self.original_tracks["video"])
            monitoring_tracks.append(video_track)
            
        if "audio" in self.original_tracks:
            monitoring_tracks.append(self.original_tracks["audio"])
            
        return monitoring_tracks

    async def create_monitoring_peer_connection(self) -> RTCPeerConnection:
        """Create a monitoring peer connection with the original streams."""
        pc = RTCPeerConnection()
        
        # Add tracks to the monitoring peer connection
        monitoring_tracks = await self.create_monitoring_tracks()
        for track in monitoring_tracks:
            pc.addTrack(track)
            
        return pc

    async def start_recording(self):
        """Start recording media streams."""
        if self.current_recorder:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.temp_dir, 
            f"recording_{self.session_id}_{timestamp}.mp4"
        )
        
        try:
            self.current_recorder = MediaRecorder(
                output_path,
                format="mp4"
            )
            
            # Add all active tracks to the recorder
            for track in self.active_tracks:
                self.current_recorder.addTrack(track)
                
            await self.current_recorder.start()
            logger.info(f"Started recording to {output_path}")
            
        except Exception as e:
            logger.error(f"Error starting recorder: {e}")
            self.current_recorder = None
        
    async def stop_recording(self):
        """Stop current recording if any."""
        if self.current_recorder:
            try:
                # First, stop adding new data
                if hasattr(self.current_recorder, '_tracks'):
                    self.current_recorder._tracks = []

                # Try to stop gracefully
                try:
                    await self.current_recorder.stop()
                except Exception as e:
                    logger.warning(f"Error during graceful stop: {e}")

                # Force cleanup
                if hasattr(self.current_recorder, '_container'):
                    try:
                        self.current_recorder._container.close()
                    except Exception as e:
                        logger.warning(f"Error closing container: {e}")

                # Clear the recorder
                self.current_recorder = None
                
                # Wait a moment for file operations to complete
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Error stopping recorder: {e}")
                self.current_recorder = None
