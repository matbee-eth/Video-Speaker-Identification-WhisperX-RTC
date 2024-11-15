from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass
class MediaFrame:
    timestamp: float
    video_frame: Optional[np.ndarray] = None
    audio_data: Optional[np.ndarray] = None