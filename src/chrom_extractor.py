"""
CHROM (Chrominance-based) Signal Extractor
Implements the CHROM algorithm for robust rPPG signal extraction
Uses chrominance-based approach with quality-weighted multi-ROI fusion
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ROIQuality:
    """Quality metrics for an individual ROI"""

    brightness: float
    contrast: float
    variance: float
    overall: float


class CHROMExtractor:
    """
    Extracts pulse signal using chrominance-based (CHROM) method
    More robust to illumination changes and motion artifacts

    Algorithm:
    1. Normalize RGB channels temporally
    2. Compute chrominance signals:
       Xs = 3*Rn - 2*Gn
       Ys = 1.5*Rn + Gn - 1.5*Bn
    3. Apply bandpass filtering to each
    4. Calculate adaptive weighting: alpha = std(Xs) / std(Ys)
    5. Pulse signal: S = Xs - alpha * Ys
    """

    def __init__(self, config):
        """
        Initialize CHROM extractor with configuration

        Args:
            config: Configuration object containing algorithm parameters
        """
        self.config = config
        self.roi_config = config.algorithm.roi
        self.skin_config = self.roi_config.skin_segmentation

        # Extract ROI information
        self.roi_regions = [r.name for r in self.roi_config.regions]
        self.roi_weights = np.array([r.weight for r in self.roi_config.regions])

        # Normalize weights
        self.roi_weights = self.roi_weights / np.sum(self.roi_weights)

        # RGB buffer for temporal normalization (per ROI)
        self.rgb_buffers: Dict[str, List[np.ndarray]] = {
            roi: [] for roi in self.roi_regions
        }

        # Quality history for adaptive weighting
        self.quality_history: Dict[str, List[ROIQuality]] = {
            roi: [] for roi in self.roi_regions
        }

        self.buffer_max_size = 300  # ~10 seconds at 30 FPS

        logger.info(f"CHROMExtractor initialized")
        logger.info(f"ROI regions: {self.roi_regions}")
        logger.info(f"ROI weights: {dict(zip(self.roi_regions, self.roi_weights))}")

    def extract_spatial_rgb(
        self, roi: np.ndarray, skin_mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Calculate spatial average of RGB values in ROI

        Args:
            roi: Input ROI in BGR format (OpenCV)
            skin_mask: Optional binary mask for skin pixels

        Returns:
            rgb_mean: Mean RGB values [R, G, B] or None if invalid
        """
        if roi is None or roi.size == 0:
            return None

        if len(roi.shape) != 3 or roi.shape[2] != 3:
            logger.warning(f"Invalid ROI shape: {roi.shape}")
            return None

        try:
            # Convert BGR to RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            if skin_mask is not None and skin_mask.size > 0:
                # Apply skin mask
                valid_pixels = np.sum(skin_mask > 0)

                if valid_pixels == 0:
                    # Fallback to full ROI
                    rgb_mean = np.mean(roi_rgb, axis=(0, 1))
                else:
                    # Mask and compute mean
                    mask_3d = np.repeat(skin_mask[:, :, np.newaxis], 3, axis=2)
                    masked_roi = roi_rgb * (mask_3d > 0)
                    rgb_sum = np.sum(masked_roi, axis=(0, 1))
                    rgb_mean = rgb_sum / valid_pixels
            else:
                # Use full ROI
                rgb_mean = np.mean(roi_rgb, axis=(0, 1))

            return rgb_mean.astype(np.float32)

        except Exception as e:
            logger.error(f"Error in extract_spatial_rgb: {e}")
            return None

    def compute_roi_quality(self, roi: np.ndarray) -> ROIQuality:
        """
        Compute quality metrics for ROI

        Args:
            roi: Input ROI image

        Returns:
            ROIQuality object with metrics
        """
        if roi is None or roi.size == 0:
            return ROIQuality(0.0, 0.0, 0.0, 0.0)

        try:
            # Convert to grayscale for quality metrics
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Brightness: mean intensity (normalized)
            brightness = np.mean(gray) / 255.0

            # Contrast: standard deviation (normalized)
            contrast = np.std(gray) / 128.0

            # Variance: coefficient of variation
            mean_val = np.mean(gray)
            if mean_val > 0:
                variance = np.std(gray) / mean_val
            else:
                variance = 0.0

            # Overall quality (weighted combination)
            # Prefer medium brightness (0.3-0.7), high contrast (>0.3)
            brightness_score = 1.0 - abs(brightness - 0.5) * 2.0
            brightness_score = max(0.0, min(1.0, brightness_score))

            contrast_score = min(1.0, contrast / 0.3)

            variance_score = min(1.0, variance / 0.5)

            overall = (
                0.4 * brightness_score + 0.4 * contrast_score + 0.2 * variance_score
            )

            return ROIQuality(
                brightness=brightness,
                contrast=contrast,
                variance=variance,
                overall=overall,
            )

        except Exception as e:
            logger.error(f"Error computing ROI quality: {e}")
            return ROIQuality(0.0, 0.0, 0.0, 0.0)

    def update_rgb_buffer(self, roi_name: str, rgb_mean: np.ndarray):
        """
        Update RGB buffer for temporal normalization

        Args:
            roi_name: Name of the ROI
            rgb_mean: Mean RGB values
        """
        buffer = self.rgb_buffers[roi_name]
        buffer.append(rgb_mean)

        # Keep buffer size limited
        if len(buffer) > self.buffer_max_size:
            buffer.pop(0)

    def extract_chrom_signal(
        self, roi_name: str, min_buffer_size: int = 30
    ) -> Optional[float]:
        """
        Extract CHROM signal from RGB buffer

        Args:
            roi_name: Name of the ROI
            min_buffer_size: Minimum buffer size required

        Returns:
            Pulse signal value or None
        """
        buffer = self.rgb_buffers[roi_name]

        if len(buffer) < min_buffer_size:
            return None

        try:
            # Convert buffer to array: (frames, 3)
            rgb_array = np.array(buffer)

            # Temporal normalization (mean normalization only - NOT z-score)
            # CHROM paper: Rn(t) = R(t) / mean(R), NOT z-score normalization
            rgb_normalized = np.zeros_like(rgb_array)
            for i in range(3):
                channel = rgb_array[:, i]
                mean = np.mean(channel)

                if mean > 1e-6:
                    rgb_normalized[:, i] = channel / mean
                else:
                    rgb_normalized[:, i] = 1.0

            # Extract normalized R, G, B
            Rn = rgb_normalized[:, 0]
            Gn = rgb_normalized[:, 1]
            Bn = rgb_normalized[:, 2]

            # Compute chrominance signals
            Xs = 3.0 * Rn - 2.0 * Gn
            Ys = 1.5 * Rn + Gn - 1.5 * Bn

            # Compute adaptive weighting
            std_Xs = np.std(Xs)
            std_Ys = np.std(Ys)

            if std_Ys > 1e-6:
                alpha = std_Xs / std_Ys
            else:
                alpha = 1.0

            # Final pulse signal
            S = Xs - alpha * Ys

            # Return latest value
            return S[-1]

        except Exception as e:
            logger.error(f"Error extracting CHROM signal for {roi_name}: {e}")
            return None

    def fuse_multi_roi_signals(
        self, roi_signals: Dict[str, float], roi_qualities: Dict[str, ROIQuality]
    ) -> Optional[float]:
        """
        Fuse signals from multiple ROIs with quality-based weighting

        Args:
            roi_signals: Dictionary of ROI name -> signal value
            roi_qualities: Dictionary of ROI name -> quality metrics

        Returns:
            Fused signal value or None
        """
        if not roi_signals:
            return None

        try:
            # Compute quality-weighted fusion
            weighted_sum = 0.0
            total_weight = 0.0

            for roi_name, signal in roi_signals.items():
                if signal is None:
                    continue

                # Base weight from configuration
                roi_idx = self.roi_regions.index(roi_name)
                base_weight = self.roi_weights[roi_idx]

                # Quality weight
                quality = roi_qualities.get(roi_name, ROIQuality(0, 0, 0, 0))
                quality_weight = quality.overall

                # Combined weight
                weight = base_weight * (0.5 + 0.5 * quality_weight)

                weighted_sum += signal * weight
                total_weight += weight

            if total_weight > 1e-6:
                fused_signal = weighted_sum / total_weight
                return fused_signal
            else:
                return None

        except Exception as e:
            logger.error(f"Error fusing multi-ROI signals: {e}")
            return None

    def process_frame_rois(
        self, rois: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]
    ) -> Optional[float]:
        """
        Process multiple ROIs from a frame and extract fused signal

        Args:
            rois: Dictionary of ROI name -> (roi_image, skin_mask)

        Returns:
            Fused pulse signal value or None
        """
        roi_signals = {}
        roi_qualities = {}

        for roi_name, (roi_img, skin_mask) in rois.items():
            if roi_img is None:
                continue

            # Extract spatial RGB mean
            rgb_mean = self.extract_spatial_rgb(roi_img, skin_mask)
            if rgb_mean is None:
                continue

            # Update buffer
            self.update_rgb_buffer(roi_name, rgb_mean)

            # Extract CHROM signal
            signal = self.extract_chrom_signal(roi_name)
            if signal is not None:
                roi_signals[roi_name] = signal

            # Compute quality
            quality = self.compute_roi_quality(roi_img)
            roi_qualities[roi_name] = quality

        # Fuse signals
        fused_signal = self.fuse_multi_roi_signals(roi_signals, roi_qualities)

        return fused_signal

    def reset(self):
        """Reset all buffers and history"""
        for roi in self.roi_regions:
            self.rgb_buffers[roi].clear()
            self.quality_history[roi].clear()

        logger.info("CHROMExtractor buffers reset")

    def get_buffer_info(self) -> Dict[str, int]:
        """Get current buffer sizes for each ROI"""
        return {roi: len(self.rgb_buffers[roi]) for roi in self.roi_regions}
