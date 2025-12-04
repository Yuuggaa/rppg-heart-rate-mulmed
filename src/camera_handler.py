"""
Async Camera Handler Module
Async context manager for camera initialization with adaptive settings
"""

import cv2
import asyncio
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class AsyncCameraHandler:
    """
    Async context manager for camera operations
    Handles camera initialization, frame capture, and reconnection
    """
    
    def __init__(self, config):
        self.config = config
        self.hardware_config = config.hardware
        self.camera_id = self.hardware_config.camera_id
        self.target_fps = self.hardware_config.target_fps
        self.resolution = (
            self.hardware_config.resolution.width,
            self.hardware_config.resolution.height
        )
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_fps: float = 0
        self.actual_resolution: Tuple[int, int] = (0, 0)
        self.is_opened: bool = False
        
        # Reconnection
        self.reconnect_config = self.hardware_config.reconnect
        self.reconnect_attempts = 0
        
        logger.info(f"AsyncCameraHandler initialized for camera {self.camera_id}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.release()
        return False
    
    async def initialize(self) -> bool:
        """
        Initialize camera with adaptive settings
        
        Returns:
            Success status
        """
        logger.info(f"Initializing camera {self.camera_id}...")
        
        # Run blocking camera init in executor
        loop = asyncio.get_event_loop()
        self.cap = await loop.run_in_executor(
            None, cv2.VideoCapture, self.camera_id
        )
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Get actual settings
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.actual_resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        
        self.is_opened = True
        
        logger.info(f"Camera initialized: {self.actual_resolution[0]}x{self.actual_resolution[1]} @ {self.actual_fps} FPS")
        
        # Warm up
        await self._warmup()
        
        return True
    
    async def _warmup(self, frames: int = 5):
        """Warm up camera by reading initial frames"""
        loop = asyncio.get_event_loop()
        for _ in range(frames):
            await loop.run_in_executor(None, self.cap.read)
            await asyncio.sleep(0.01)
    
    async def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from camera asynchronously
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        try:
            loop = asyncio.get_event_loop()
            ret, frame = await loop.run_in_executor(None, self.cap.read)
            
            if not ret:
                logger.warning("Failed to read frame from camera")
                
                # Try reconnection
                if await self._should_reconnect():
                    await self._reconnect()
                    return await self.read_frame()
            
            return ret, frame
        
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None
    
    def _should_reconnect(self) -> bool:
        """Check if should attempt reconnection"""
        return (self.reconnect_attempts < 
                self.reconnect_config.max_retries)
    
    async def _reconnect(self):
        """Attempt to reconnect camera"""
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts > self.reconnect_config.max_retries:
            logger.error("Max reconnection attempts reached")
            return
        
        # Get wait time with exponential backoff
        wait_idx = min(self.reconnect_attempts - 1, 
                      len(self.reconnect_config.backoff_intervals) - 1)
        wait_time = self.reconnect_config.backoff_intervals[wait_idx]
        
        logger.info(f"Reconnection attempt {self.reconnect_attempts}, waiting {wait_time}s...")
        await asyncio.sleep(wait_time)
        
        # Release old capture
        if self.cap is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.cap.release)
        
        # Reinitialize
        success = await self.initialize()
        
        if success:
            logger.info("Camera reconnection successful")
            self.reconnect_attempts = 0
        else:
            logger.error("Camera reconnection failed")
    
    async def release(self):
        """Release camera resources"""
        if self.cap is not None:
            logger.info("Releasing camera...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.cap.release)
            self.cap = None
            self.is_opened = False
    
    def get_fps(self) -> float:
        """Get actual camera FPS"""
        return self.actual_fps
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get actual camera resolution"""
        return self.actual_resolution
    
    def is_available(self) -> bool:
        """Check if camera is available"""
        return self.is_opened and self.cap is not None and self.cap.isOpened()
