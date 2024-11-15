import asyncio
from aiortc import AudioStreamTrack
import logging
import numpy as np
from media_frame import MediaFrame
# import MediaStreamError
from aiortc.contrib.media import MediaStreamError
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessedAudioTrack(AudioStreamTrack):
    def __init__(self, track, processor):
        super().__init__()
        self.track = track
        self.processor = processor
        self._started = False
        self._queue = asyncio.Queue()
        self._task = None
        self.frame_count = 0
        self.sample_rate = None
        self.channels = None

    async def _process_queue(self):
        while True:
            try:
                frame = await self._queue.get()
                
                # Get raw audio data using frame's buffer
                audio_buffer = bytes(frame.planes[0])
                audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
                
                # Handle multi-channel audio if needed
                num_channels = len(frame.layout.channels)
                if num_channels > 1:
                    audio_data = audio_data.reshape(-1, num_channels)
                    audio_data = audio_data.mean(axis=1).astype(np.int16)

                # Add to processing queue
                self.processor.queue_processor.add_frame(MediaFrame(
                    timestamp=frame.time,
                    audio_data=audio_data
                ))
                
                logger.info(f"Processed audio frame: samples={len(audio_data)}, "
                          f"range=[{audio_data.min()}, {audio_data.max()}], "
                          f"time={frame.time:.3f}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing audio frame: {e}", exc_info=True)

    async def recv(self):
        if not self._started:
            await self.start()
        
        try:
            frame = await self.track.recv()
            
            # Initialize audio parameters from first frame
            if self.sample_rate is None:
                self.sample_rate = frame.sample_rate
                self.channels = len(frame.layout.channels)
                logger.info(f"Initialized audio: {self.sample_rate}Hz, {self.channels} channels")

            # Log frame details periodically
            if self.frame_count % 100 == 0:
                audio_buffer = bytes(frame.planes[0])
                audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
                logger.info(f"Audio frame {self.frame_count}: samples={len(audio_data)}")
            
            # Queue frame for processing
            await self._queue.put(frame)
            self.frame_count += 1
            
            return frame
            
        except MediaStreamError:
            logger.info("Media stream ended")
            self._ended = True
            if self._task:
                self._task.cancel()
            raise
        except Exception as e:
            logger.error(f"Error in recv: {e}", exc_info=True)
            raise

    async def start(self):
        if self._started:
            return
        self._started = True
        self._task = asyncio.create_task(self._process_queue())
        
    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._started = False
        super().stop()
