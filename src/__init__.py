"""
rPPG Heart Rate Monitoring System
Real-time remote photoplethysmography using webcam
"""

__version__ = "1.0.0"
__author__ = "rPPG System"

# Make modules accessible
from . import config
from . import utils
from . import camera_handler
from . import face_detector
from . import lighting_handler
from . import motion_detector
from . import pos_extractor
from . import signal_processor
from . import visualizer

__all__ = [
    "config",
    "utils",
    "camera_handler",
    "face_detector",
    "lighting_handler",
    "motion_detector",
    "pos_extractor",
    "signal_processor",
    "visualizer",
]
