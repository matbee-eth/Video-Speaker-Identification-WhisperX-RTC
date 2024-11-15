import asyncio
from asyncio import subprocess
from typing import Dict
import wave
from fastapi import FastAPI, WebSocket
import logging
import json
import os
from datetime import datetime
import cv2
import numpy as np
import torch
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    RTCConfiguration,
    RTCIceServer,
)
from media_stream_processor import MediaStreamProcessor
from enhanced_speaker_detection import EnhancedSpeakerDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RTCServer:
    def __init__(self):
        self.app = FastAPI()
        self.connections: Dict[str, RTCPeerConnection] = {}
        self.monitoring_connections: Dict[str, RTCPeerConnection] = {}
        self.processors: Dict[str, MediaStreamProcessor] = {}
        
        # Create output directory for saved recordings
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.rtc_config = RTCConfiguration([
            RTCIceServer(urls=["stun:stun.l.google.com:19302"])
        ])
        
        @self.app.websocket("/rtc")
        async def websocket_rtc(websocket: WebSocket):
            await self.handle_connection(websocket)
            
        @self.app.websocket("/monitor")
        async def websocket_monitor(websocket: WebSocket):
            await self.handle_monitoring(websocket)

    async def save_processed_recording(self, session_id: str):
        """Save the processed recording with speaker detection and subtitles."""
        try:
            processor = self.processors.get(session_id)
            if not processor:
                logger.error("No processor found for session")
                return

            logger.info("Starting post-processing of recording")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"processed_recording_{session_id}_{timestamp}.mp4")

            # Get the recording path
            recording_files = [f for f in os.listdir(processor.temp_dir) 
                            if f.endswith('.mp4') and os.path.getsize(os.path.join(processor.temp_dir, f)) > 0]
            
            if not recording_files:
                logger.error(f"No valid recording found for session {session_id}")
                return

            temp_path = os.path.join(processor.temp_dir, recording_files[0])
            logger.info(f"Found recording at: {temp_path}")
            
            # Verify the file is a valid video file
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                logger.error(f"Invalid video file: {temp_path}")
                return

            # Get video info
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")

            # Create video writer
            temp_output = f"{output_path}.temp.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            if not out.isOpened():
                logger.error("Failed to create output video file")
                return

            # Get complete audio data and save it to a temporary WAV file
            audio_data = processor.queue_processor.audio_buffer.get_complete_audio()
            if audio_data is None or len(audio_data) == 0:
                logger.error("No audio data available for processing")
                return
            
            logger.info(f"Audio data: {len(audio_data)} samples")

            # Save audio data to temporary WAV file
            temp_audio_path = os.path.join(processor.temp_dir, f"temp_audio_{timestamp}.wav")
            with wave.open(temp_audio_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # WhisperX expects 16kHz
                wav_file.writeframes(audio_data.tobytes())

            # Collect frames
            frames = []
            frame_timestamps = []
            frame_count = 0
            
            logger.info("Reading frames...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                timestamp = frame_count / fps
                frames.append(frame)
                frame_timestamps.append(timestamp)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Read {frame_count}/{total_frames} frames")

            cap.release()
            logger.info(f"Read {len(frames)} frames")

            # Process with speaker detection
            if frames and processor.speaker_detector:
                try:
                    logger.info("Starting speaker detection processing")
                    segments, face_detections = processor.speaker_detector.process_video_segment(
                        temp_audio_path,  # Now passing the path to the WAV file
                        list(zip(frame_timestamps, frames))
                    )
                    
                    if segments:
                        logger.info(f"Found {len(segments)} speaker segments")
                    else:
                        logger.warning("No speaker segments detected")

                    if face_detections:
                        logger.info(f"Processing {len(face_detections)} face detections")
                    else:
                        logger.warning("No face detections found")

                    # Write processed frames
                    logger.info("Writing processed frames...")
                    for i, (timestamp, frame) in enumerate(zip(frame_timestamps, frames)):
                        current_detections = []
                        if face_detections and i < len(face_detections):
                            _, detections = face_detections[i]
                            current_detections = detections

                        current_segments = []
                        if segments:
                            current_segments = [
                                seg for seg in segments
                                if seg["start"] <= timestamp <= seg["end"]
                            ]

                        processed_frame = self._draw_visualization(
                            frame,
                            current_detections,
                            current_segments,
                            height,
                            width
                        )
                        
                        out.write(processed_frame)
                        
                        if i % 100 == 0:
                            logger.info(f"Processed {i}/{len(frames)} frames")

                except Exception as e:
                    logger.error(f"Error in speaker detection processing: {e}", exc_info=True)
                    logger.info("Falling back to original frames")
                    for frame in frames:
                        out.write(frame)

                finally:
                    # Clean up the temporary audio file
                    try:
                        os.remove(temp_audio_path)
                    except Exception as e:
                        logger.warning(f"Error removing temporary audio file: {e}")

            out.release()
            logger.info("Video processing complete, combining with audio...")

            # Combine processed video with original audio using asyncio subprocess
            try:
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', temp_output,     # Processed video
                    '-i', temp_path,       # Original video (for audio)
                    '-c:v', 'libx264',     # Video codec
                    '-preset', 'medium',    # Encoding preset
                    '-crf', '23',          # Quality
                    '-c:a', 'aac',         # Audio codec
                    '-b:a', '128k',        # Audio bitrate
                    '-map', '0:v:0',       # Use video from first input
                    '-map', '1:a:0',       # Use audio from second input
                    '-y',                  # Overwrite output
                    output_path
                ]
                
                logger.info("Running FFmpeg for final encoding")
                process = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"FFmpeg error: {stderr.decode()}")
                else:
                    logger.info("FFmpeg encoding complete")
                    
                # Cleanup temp file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    
                logger.info(f"Saved processed recording to {output_path}")
                
            except Exception as e:
                logger.error(f"Error in FFmpeg processing: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error saving processed recording: {e}", exc_info=True)

    async def finalize_recording(self, session_id: str):
        """Finalize the recording and process it."""
        try:
            logger.info(f"Finalizing recording for session {session_id}")
            processor = self.processors.get(session_id)
            if not processor:
                return

            # Stop recording first
            await processor.stop_recording()
            await asyncio.sleep(1)  # Give time for recording to finish

            # Process any remaining data in the queue
            if processor.queue_processor:
                processor.queue_processor._running = False
                await processor.queue_processor._process_batch()
                await asyncio.sleep(0.5)  # Give time for processing to complete

            # Save the processed recording
            await self.save_processed_recording(session_id)

        except Exception as e:
            logger.error(f"Error finalizing recording: {e}", exc_info=True)


    def _draw_visualization(self, frame: np.ndarray, detections: list, 
                          segments: list, height: int, width: int) -> np.ndarray:
        """Draw speaker detection visualization on frame."""
        output = frame.copy()

        # Draw face detections
        for detection in detections:
            bbox = detection["bbox"]
            is_speaking = detection["is_speaking"]
            speaker_id = detection.get("speaker_id")
            confidence = detection.get("speaker_confidence", 0)

            # Choose color based on speaker status
            if speaker_id is not None:
                # Generate consistent color for speaker
                hue = (speaker_id * 40) % 180
                color = tuple(int(x) for x in cv2.cvtColor(
                    np.uint8([[[hue, 255, 255]]]),
                    cv2.COLOR_HSV2BGR)[0][0]
                )
            else:
                color = (0, 255, 0) if is_speaking else (0, 0, 255)

            # Draw bounding box
            x, y, w, h = map(int, bbox)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            # Draw label background
            label = f"Speaker {speaker_id}" if speaker_id is not None else (
                "Speaking" if is_speaking else "Silent"
            )
            cv2.rectangle(output, (x, y - 30), (x + w, y), color, -1)
            
            # Draw label text
            cv2.putText(output, label, (x + 5, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if speaker_id is not None:
                conf_text = f"Conf: {confidence:.1%}"
                cv2.putText(output, conf_text, (x + 5, y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw subtitles
        if segments:
            text_lines = []
            for segment in segments:
                speaker_id = segment.get("speaker")
                text = segment.get("text", "")
                if speaker_id is not None and text:
                    text_lines.append(f"Speaker {speaker_id}: {text}")

            if text_lines:
                text = " | ".join(text_lines)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                
                (text_width, text_height), _ = cv2.getTextSize(
                    text, font, font_scale, thickness
                )
                
                text_x = (width - text_width) // 2
                text_y = height - 30
                
                # Draw background rectangle
                padding = 10
                cv2.rectangle(
                    output,
                    (text_x - padding, text_y - text_height - padding),
                    (text_x + text_width + padding, text_y + padding),
                    (0, 0, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    output, text, (text_x, text_y),
                    font, font_scale, (255, 255, 255), thickness
                )

        return output


    async def handle_connection(self, websocket: WebSocket):
        await websocket.accept()
        session_id = None
        pc = None
        processor = None
        
        try:
            while True:
                message = await websocket.receive_json()
                logger.info(f"Received message type: {message.get('type')}")

                if message["type"] == "offer":
                    session_id = str(len(self.connections))
                    pc = RTCPeerConnection(configuration=self.rtc_config)
                    
                    # Create processor with integrated speaker detection
                    processor = MediaStreamProcessor(session_id)
                    processor.speaker_detector = EnhancedSpeakerDetector(
                        min_speakers=1,
                        max_speakers=4,
                        temporal_window=10,
                        lip_movement_threshold=0.15
                    )
                    
                    self.connections[session_id] = pc
                    self.processors[session_id] = processor
                    
                    # Create data channel for detection results
                    data_channel = pc.createDataChannel("detections")
                    processor.set_data_channel(data_channel)
                    
                    @data_channel.on("open")
                    def on_open():
                        logger.info("Data channel opened")
                        
                    @data_channel.on("close")
                    async def on_close():
                        logger.info("Data channel closed")
                        # Ensure we process any remaining data when the channel closes
                        if processor and processor.queue_processor:
                            await processor.queue_processor._process_batch()
                        
                    @pc.on("connectionstatechange")
                    async def on_connectionstatechange():
                        logger.info(f"Connection state is {pc.connectionState}")
                        if pc.connectionState == "connected":
                            await processor.start_recording()
                        elif pc.connectionState in ["failed", "closed"]:
                            # Process any remaining data before cleanup
                            if processor and processor.queue_processor:
                                await processor.queue_processor._process_batch()
                            await self.finalize_recording(session_id)
                            await self.cleanup_connection(session_id)
                    
                    @pc.on("track")
                    async def on_track(track):
                        logger.info(f"Received {track.kind} track")
                        try:
                            relayed_track = await processor.add_track(track, pc)
                            pc.addTrack(relayed_track)
                            
                            @track.on("ended")
                            async def on_ended():
                                logger.info(f"Track {track.kind} ended")
                                # If both tracks have ended, finalize the recording
                                if all(t.readyState == "ended" for t in pc.getTransceivers()):
                                    await self.finalize_recording(session_id)
                                
                        except Exception as e:
                            logger.error(f"Error processing track: {e}")
                    
                    # Set remote description
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=message["sdp"], type=message["type"])
                    )
                    
                    # Create and send answer
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    
                    await websocket.send_json({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp,
                        "session_id": session_id
                    })

                elif message["type"] == "stop" and session_id:
                    # Handle explicit stop message from client
                    logger.info("Received stop message")
                    await self.finalize_recording(session_id)
                    await self.cleanup_connection(session_id)
                    break

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if session_id:
                await self.cleanup_connection(session_id)
            await websocket.close()

    async def cleanup_connection(self, session_id: str):
        """Clean up resources."""
        try:
            if session_id in self.processors:
                processor = self.processors[session_id]
                
                # Stop queue processor first
                if processor.queue_processor:
                    processor.queue_processor._running = False
                    await asyncio.sleep(0.5)  # Give time for processor to stop
                
                # Then cleanup processor
                await processor.cleanup()
                del self.processors[session_id]
            
            # Cleanup connections
            if session_id in self.connections:
                pc = self.connections[session_id]
                await pc.close()
                del self.connections[session_id]
                
            if session_id in self.monitoring_connections:
                pc = self.monitoring_connections[session_id]
                await pc.close()
                del self.monitoring_connections[session_id]
                
        except Exception as e:
            logger.error(f"Error in cleanup: {e}", exc_info=True)
