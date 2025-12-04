"""
Modern Lighting Handler with Gamma Correction
Handles lighting preprocessing with gamma correction and auto-adjustment
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class LightingHandler:
    """
    Handles lighting normalization with gamma correction and change detection
    """
    
    def __init__(self, config):
        self.config = config
        self.lighting_config = config.lighting
        
        # Gamma correction settings
        self.gamma_config = self.lighting_config.gamma
        self.current_gamma = self.gamma_config.default_value
        self.gamma_range = self.gamma_config.range
        
        # CLAHE settings
        self.clahe_config = self.lighting_config.clahe
        if self.clahe_config.enabled:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_config.clip_limit,
                tileGridSize=tuple(self.clahe_config.tile_size)
            )
        else:
            self.clahe = None
        
        # Change detection
        self.change_config = self.lighting_config.change_detection
        self.brightness_history = deque(maxlen=self.change_config.history_size)
        self.frames_since_change = 0
        self.change_detected = False
        
        # Build gamma lookup table
        self._build_gamma_table()
        
        logger.info(f"LightingHandler initialized")
        logger.info(f"Gamma: {self.current_gamma}, Auto-adjust: {self.gamma_config.auto_adjust}")
        logger.info(f"CLAHE: {self.clahe_config.enabled}")
    
    def _build_gamma_table(self):
        """Build lookup table for gamma correction"""
        inv_gamma = 1.0 / self.current_gamma
        self.gamma_table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)
    
    def apply_gamma_correction(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction to frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Gamma-corrected frame
        """
        if not self.gamma_config.enabled:
            return frame
        
        return cv2.LUT(frame, self.gamma_table)
    
    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            frame: Input BGR frame
            
        Returns:
            CLAHE-enhanced frame
        """
        if not self.clahe_config.enabled or self.clahe is None:
            return frame
        
        # Convert to YUV
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        # Apply CLAHE to Y channel
        yuv[:, :, 0] = self.clahe.apply(yuv[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        return enhanced
    
    def calculate_brightness(self, frame: np.ndarray) -> float:
        """
        Calculate average brightness of frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Average brightness (0-255)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def auto_adjust_gamma(self, frame: np.ndarray):
        """
        Automatically adjust gamma based on frame brightness
        
        Args:
            frame: Input BGR frame
        """
        if not self.gamma_config.auto_adjust:
            return
        
        brightness = self.calculate_brightness(frame)
        
        # Target brightness is around 128 (middle gray)
        target = 128.0
        
        # Calculate adjustment
        if brightness < target - 30:
            # Too dark, increase gamma
            self.current_gamma = min(
                self.gamma_range[1],
                self.current_gamma + 0.05
            )
        elif brightness > target + 30:
            # Too bright, decrease gamma
            self.current_gamma = max(
                self.gamma_range[0],
                self.current_gamma - 0.05
            )
        
        # Rebuild lookup table
        self._build_gamma_table()
    
    def detect_lighting_change(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect sudden lighting changes
        
        Args:
            frame: Current BGR frame
            
        Returns:
            Tuple of (change_detected, change_magnitude)
        """
        if not self.change_config.enabled:
            return False, 0.0
        
        current_brightness = self.calculate_brightness(frame)
        
        # Add to history
        self.brightness_history.append(current_brightness)
        
        # Need at least 2 frames
        if len(self.brightness_history) < 2:
            return False, 0.0
        
        # Calculate change from recent average
        if len(self.brightness_history) >= 5:
            recent_avg = np.mean(list(self.brightness_history)[-5:-1])
            change = abs(current_brightness - recent_avg)
            
            if change > self.change_config.threshold:
                self.change_detected = True
                self.frames_since_change = 0
                logger.warning(f"Lighting change detected: {change:.1f}")
                return True, change
        
        # Update change status
        if self.change_detected:
            self.frames_since_change += 1
            grace_frames = int(self.change_config.grace_period * 30)  # Assume 30 FPS
            
            if self.frames_since_change >= grace_frames:
                self.change_detected = False
                logger.info("Lighting stabilized")
        
        return False, 0.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process frame with all lighting enhancements
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (processed_frame, lighting_stable)
        """
        # Auto-adjust gamma
        self.auto_adjust_gamma(frame)
        
        # Apply gamma correction
        processed = self.apply_gamma_correction(frame)
        
        # Apply CLAHE
        processed = self.apply_clahe(processed)
        
        # Detect changes
        change_detected, _ = self.detect_lighting_change(frame)
        lighting_stable = not change_detected
        
        return processed, lighting_stable
    
    def is_lighting_stable(self) -> bool:
        """Check if lighting is currently stable"""
        return not self.change_detected
    
    def reset(self):
        """Reset brightness history"""
        self.brightness_history.clear()
        self.change_detected = False
        self.frames_since_change = 0
        logger.info("LightingHandler reset")
