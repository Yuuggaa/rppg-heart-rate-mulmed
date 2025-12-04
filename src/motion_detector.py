"""
Modern Motion Detector with Lucas-Kanade Sparse Optical Flow
Implements sparse optical flow for efficient motion detection
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class MotionDetector:
    """
    Detects motion using Lucas-Kanade sparse optical flow
    """
    
    def __init__(self, config):
        self.config = config
        self.motion_config = config.motion
        
        # Sparse LK parameters
        self.lk_config = self.motion_config.sparse_lk
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=self.lk_config.feature_params.max_corners,
            qualityLevel=self.lk_config.feature_params.quality_level,
            minDistance=self.lk_config.feature_params.min_distance,
            blockSize=self.lk_config.feature_params.block_size
        )
        
        self.threshold = self.lk_config.threshold
        
        # Variance checking
        self.variance_config = self.motion_config.variance
        self.signal_buffer = deque(maxlen=self.variance_config.buffer_size)
        
        # State
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.motion_detected = False
        self.motion_magnitude = 0.0
        self.grace_counter = 0
        self.grace_period_frames = int(self.motion_config.grace_period * 30)  # Assume 30 FPS
        
        logger.info(f"MotionDetector initialized with Lucas-Kanade sparse optical flow")
        logger.info(f"Threshold: {self.threshold} pixels")
    
    def _detect_features(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Detect good features to track
        
        Args:
            gray: Grayscale image
            mask: Optional mask to restrict feature detection
            
        Returns:
            Feature points or None
        """
        try:
            points = cv2.goodFeaturesToTrack(
                gray, 
                mask=mask,
                **self.feature_params
            )
            return points
        except Exception as e:
            logger.error(f"Feature detection error: {e}")
            return None
    
    def detect_motion_optical_flow(self, frame: np.ndarray, roi_bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[float, bool]:
        """
        Detect motion using sparse Lucas-Kanade optical flow
        
        Args:
            frame: Current BGR frame
            roi_bbox: Optional ROI (x, y, w, h) to focus on
            
        Returns:
            Tuple of (motion_magnitude, motion_detected)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create mask for ROI if specified
        mask = None
        if roi_bbox is not None:
            x, y, w, h = roi_bbox
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
            gray_roi = gray.copy()
        else:
            gray_roi = gray
        
        # Initialize or update features
        if self.prev_gray is None or self.prev_points is None:
            self.prev_gray = gray
            self.prev_points = self._detect_features(gray_roi, mask)
            return 0.0, False
        
        # Need valid previous points
        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_gray = gray
            self.prev_points = self._detect_features(gray_roi, mask)
            return 0.0, False
        
        try:
            # Calculate optical flow
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.prev_points,
                None,
                **self.lk_params
            )
            
            # Select good points
            if next_points is not None and status is not None:
                good_prev = self.prev_points[status == 1]
                good_next = next_points[status == 1]
                
                if len(good_prev) > 0 and len(good_next) > 0:
                    # Calculate displacement
                    displacement = good_next - good_prev
                    distances = np.sqrt(np.sum(displacement**2, axis=1))
                    
                    # Average motion magnitude
                    motion_magnitude = np.mean(distances)
                    
                    # Check threshold
                    motion_detected = motion_magnitude > self.threshold
                    
                    # Update state
                    self.prev_gray = gray
                    self.prev_points = self._detect_features(gray_roi, mask)
                    self.motion_magnitude = motion_magnitude
                    
                    return motion_magnitude, motion_detected
            
            # If flow calculation fails, reset
            self.prev_gray = gray
            self.prev_points = self._detect_features(gray_roi, mask)
            return 0.0, False
        
        except Exception as e:
            logger.error(f"Optical flow error: {e}")
            self.prev_gray = gray
            self.prev_points = self._detect_features(gray_roi, mask)
            return 0.0, False
    
    def detect_motion_variance(self, signal_value: float) -> bool:
        """
        Detect motion via signal variance
        
        Args:
            signal_value: Current signal value
            
        Returns:
            Motion detected status
        """
        if not self.variance_config.enabled:
            return False
        
        self.signal_buffer.append(signal_value)
        
        if len(self.signal_buffer) < 10:
            return False
        
        # Calculate coefficient of variation
        signal_array = np.array(self.signal_buffer)
        mean = np.mean(signal_array)
        
        if mean == 0:
            return False
        
        std = np.std(signal_array)
        cv = std / abs(mean)
        
        return cv > self.variance_config.threshold
    
    def update_motion_state(self, flow_detected: bool, variance_detected: bool = False) -> bool:
        """
        Update motion state with grace period
        
        Args:
            flow_detected: Motion detected from optical flow
            variance_detected: Motion detected from signal variance
            
        Returns:
            Final motion state
        """
        # Motion detected if either method detects
        current_motion = flow_detected or variance_detected
        
        if current_motion:
            self.motion_detected = True
            self.grace_counter = 0
        else:
            # Apply grace period
            if self.motion_detected:
                self.grace_counter += 1
                
                if self.grace_counter >= self.grace_period_frames:
                    self.motion_detected = False
                    self.grace_counter = 0
                    logger.info("Motion grace period ended, resuming processing")
        
        return self.motion_detected
    
    def is_motion_detected(self) -> bool:
        """Check if motion is currently detected"""
        return self.motion_detected
    
    def get_motion_magnitude(self) -> float:
        """Get current motion magnitude"""
        return self.motion_magnitude
    
    def reset(self):
        """Reset motion detector state"""
        self.prev_gray = None
        self.prev_points = None
        self.signal_buffer.clear()
        self.motion_detected = False
        self.motion_magnitude = 0.0
        self.grace_counter = 0
        logger.info("MotionDetector reset")
