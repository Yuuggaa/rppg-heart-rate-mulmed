"""
Modern Signal Processing Pipeline
Implements Chebyshev Type II filter, Welch periodogram, Savitzky-Golay smoothing,
and Peak Frequency Tracking for robust BPM estimation
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import find_peaks, cheby2, filtfilt, savgol_filter, welch
from collections import deque
import logging
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BPMEstimate:
    """Container for BPM estimation results"""

    value: float
    confidence: float
    sqi: float
    snr: float
    method: str


@dataclass
class SignalQuality:
    """Signal quality metrics"""

    sqi: float
    snr: float
    variance: float
    periodicity: float


class SignalProcessor:
    """
    Modern signal processor with Chebyshev Type II filtering,
    Welch periodogram analysis, and Peak Frequency Tracking
    """

    def __init__(self, config, fps: int):
        self.config = config
        self.fps = fps
        self.processing_config = config.processing

        # Calculate buffer sizes
        self.fast_window_size = int(fps * self.processing_config.windows.fast.duration)
        self.slow_window_size = int(fps * self.processing_config.windows.slow.duration)

        # Dual circular buffers
        self.buffer_fast = deque(maxlen=self.fast_window_size)
        self.buffer_slow = deque(maxlen=self.slow_window_size)

        # Update intervals
        self.fast_update_interval = self.processing_config.windows.fast.update_interval
        self.slow_update_interval = self.processing_config.windows.slow.update_interval

        # Timestamps
        self.last_fast_update = 0.0
        self.last_slow_update = 0.0
        self.current_time = 0.0

        # BPM estimates
        self.bpm_fast: Optional[BPMEstimate] = None
        self.bpm_slow: Optional[BPMEstimate] = None
        self.bpm_active: Optional[BPMEstimate] = None

        # EMA smoothing
        self.bpm_ema = None
        self.ema_alpha = (
            self.processing_config.smoothing.ema_alpha
            if self.processing_config.smoothing.enabled
            else 1.0
        )

        # Peak frequency tracking
        self.previous_peak_freq = None
        self.pft_config = self.processing_config.bpm.pft

        # Motion hold
        self.last_valid_bpm = None
        self.motion_hold_time = 0.0
        self.max_hold_duration = config.motion.hold_duration

        # Design filters
        self._design_filters()

        logger.info(f"SignalProcessor initialized: FPS={fps}")
        logger.info(
            f"Fast window: {self.fast_window_size} samples ({self.processing_config.windows.fast.duration}s)"
        )
        logger.info(
            f"Slow window: {self.slow_window_size} samples ({self.processing_config.windows.slow.duration}s)"
        )

    def _design_filters(self):
        """Design Chebyshev Type II bandpass filter"""
        filter_config = self.processing_config.filter

        nyquist = self.fps / 2.0
        low = filter_config.bandpass.low / nyquist
        high = filter_config.bandpass.high / nyquist

        # Clip to valid range
        low = np.clip(low, 0.01, 0.99)
        high = np.clip(high, 0.01, 0.99)

        if high <= low:
            high = low + 0.1

        try:
            # Chebyshev Type II filter - better stopband attenuation
            self.b, self.a = cheby2(
                filter_config.order, filter_config.ripple_db, [low, high], btype="band"
            )
            logger.info(
                f"Chebyshev Type II filter designed: {filter_config.bandpass.low}-{filter_config.bandpass.high} Hz"
            )
        except Exception as e:
            logger.error(f"Filter design failed: {e}")
            self.b, self.a = None, None

    def detrend_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Remove baseline wander using Savitzky-Golay filter

        Args:
            signal_data: Input signal

        Returns:
            Detrended signal
        """
        if signal_data is None or len(signal_data) < 10:
            return signal_data

        detrend_config = self.processing_config.detrending

        if detrend_config.method == "savgol":
            # Savitzky-Golay smoothing for baseline
            window_length = min(detrend_config.savgol.window_length, len(signal_data))

            # Window length must be odd
            if window_length % 2 == 0:
                window_length -= 1

            # Ensure window_length > polyorder
            polyorder = min(detrend_config.savgol.polyorder, window_length - 1)

            if window_length >= polyorder + 2:
                try:
                    baseline = savgol_filter(signal_data, window_length, polyorder)
                    return signal_data - baseline
                except Exception as e:
                    logger.warning(f"Savitzky-Golay detrending failed: {e}")
                    return signal_data

        elif detrend_config.method == "polynomial":
            # Polynomial detrending
            x = np.arange(len(signal_data))
            coeffs = np.polyfit(x, signal_data, detrend_config.polynomial.order)
            trend = np.polyval(coeffs, x)
            return signal_data - trend

        return signal_data

    def apply_bandpass_filter(self, signal_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply Chebyshev Type II bandpass filter

        Args:
            signal_data: Input signal

        Returns:
            Filtered signal or None
        """
        if signal_data is None or self.b is None or self.a is None:
            return signal_data

        # Minimum length for filtfilt
        min_length = max(len(self.b), len(self.a)) * 3

        if len(signal_data) < min_length:
            return signal_data

        try:
            filtered = filtfilt(self.b, self.a, signal_data)
            return filtered
        except Exception as e:
            logger.error(f"Bandpass filtering failed: {e}")
            return signal_data

    def estimate_bpm_welch(
        self, signal_data: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """
        Estimate BPM using Welch's periodogram method

        Args:
            signal_data: Filtered signal

        Returns:
            Tuple of (bpm, confidence)
        """
        if signal_data is None or len(signal_data) < 30:
            return None, 0.0

        welch_config = self.processing_config.bpm.welch
        bpm_config = self.processing_config.bpm

        try:
            # Compute Welch periodogram
            nperseg = min(welch_config.nperseg, len(signal_data))
            noverlap = min(welch_config.noverlap, nperseg - 1)

            freqs, power = welch(
                signal_data,
                fs=self.fps,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="spectrum",
            )

            # Convert to BPM range
            freq_min = bpm_config.min / 60.0
            freq_max = bpm_config.max / 60.0

            # Find indices in valid range
            valid_idx = (freqs >= freq_min) & (freqs <= freq_max)

            if not np.any(valid_idx):
                return None, 0.0

            freqs_valid = freqs[valid_idx]
            power_valid = power[valid_idx]

            # Apply Peak Frequency Tracking if enabled
            if self.pft_config.enabled and self.previous_peak_freq is not None:
                # Focus on region around previous peak
                tracking_range = self.pft_config.tracking_range
                lower_freq = max(freq_min, self.previous_peak_freq - tracking_range)
                upper_freq = min(freq_max, self.previous_peak_freq + tracking_range)

                tracking_idx = (freqs_valid >= lower_freq) & (freqs_valid <= upper_freq)

                if np.any(tracking_idx):
                    freqs_valid = freqs_valid[tracking_idx]
                    power_valid = power_valid[tracking_idx]

            # Find peak
            peak_idx = np.argmax(power_valid)
            peak_freq = freqs_valid[peak_idx]
            peak_power = power_valid[peak_idx]

            # Convert to BPM
            bpm = peak_freq * 60.0

            # Calculate confidence based on peak prominence
            if len(power_valid) > 1:
                mean_power = np.mean(power_valid)
                if mean_power > 0:
                    prominence = (peak_power - mean_power) / mean_power
                    confidence = min(1.0, prominence / 2.0)
                else:
                    confidence = 0.0
            else:
                confidence = 0.5

            # Update tracked peak
            self.previous_peak_freq = peak_freq

            return bpm, confidence

        except Exception as e:
            logger.error(f"Welch BPM estimation failed: {e}")
            return None, 0.0

    def estimate_bpm_fft(
        self, signal_data: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """
        Estimate BPM using FFT method (fallback)

        Args:
            signal_data: Filtered signal

        Returns:
            Tuple of (bpm, confidence)
        """
        if signal_data is None or len(signal_data) < 30:
            return None, 0.0

        bpm_config = self.processing_config.bpm

        try:
            # Compute FFT
            fft_data = np.fft.fft(signal_data)
            power = np.abs(fft_data) ** 2
            freqs = np.fft.fftfreq(len(signal_data), 1.0 / self.fps)

            # Take positive frequencies
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            power = power[pos_mask]

            # Filter to physiological range
            freq_min = bpm_config.min / 60.0
            freq_max = bpm_config.max / 60.0

            valid_idx = (freqs >= freq_min) & (freqs <= freq_max)

            if not np.any(valid_idx):
                return None, 0.0

            freqs_valid = freqs[valid_idx]
            power_valid = power[valid_idx]

            # Find peak
            peak_idx = np.argmax(power_valid)
            peak_freq = freqs_valid[peak_idx]
            peak_power = power_valid[peak_idx]

            bpm = peak_freq * 60.0

            # Confidence
            mean_power = np.mean(power_valid)
            if mean_power > 0:
                prominence = (peak_power - mean_power) / mean_power
                confidence = min(1.0, prominence / 2.0)
            else:
                confidence = 0.0

            return bpm, confidence

        except Exception as e:
            logger.error(f"FFT BPM estimation failed: {e}")
            return None, 0.0

    def calculate_signal_quality(
        self, raw_signal: np.ndarray, filtered_signal: np.ndarray
    ) -> SignalQuality:
        """
        Calculate comprehensive signal quality metrics

        Args:
            raw_signal: Raw signal data
            filtered_signal: Filtered signal data

        Returns:
            SignalQuality object
        """
        if filtered_signal is None or len(filtered_signal) < 10:
            return SignalQuality(0.0, 0.0, 0.0, 0.0)

        try:
            # SNR calculation
            signal_power = np.var(filtered_signal)
            if raw_signal is not None and len(raw_signal) == len(filtered_signal):
                noise = raw_signal - filtered_signal
                noise_power = np.var(noise)
            else:
                noise_power = signal_power * 0.1

            if noise_power > 1e-10:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = 0.0

            # Variance quality (coefficient of variation)
            mean_val = np.mean(np.abs(filtered_signal))
            if mean_val > 0:
                cv = np.std(filtered_signal) / mean_val
                variance_quality = np.exp(-cv)  # Lower CV is better
            else:
                variance_quality = 0.0

            # Periodicity (autocorrelation-based)
            if len(filtered_signal) > 20:
                autocorr = np.correlate(filtered_signal, filtered_signal, mode="full")
                autocorr = autocorr[len(autocorr) // 2 :]
                autocorr = autocorr / autocorr[0]

                # Find peaks in autocorrelation
                peaks, _ = find_peaks(autocorr[1:], height=0.3)

                if len(peaks) > 0:
                    periodicity = np.mean(autocorr[peaks + 1])
                else:
                    periodicity = 0.0
            else:
                periodicity = 0.0

            # Overall SQI (weighted combination)
            snr_normalized = np.clip(snr / 20.0, 0, 1)  # Normalize SNR
            sqi = 0.4 * snr_normalized + 0.3 * variance_quality + 0.3 * periodicity

            return SignalQuality(
                sqi=sqi, snr=snr, variance=variance_quality, periodicity=periodicity
            )

        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return SignalQuality(0.0, 0.0, 0.0, 0.0)

    def process_signal_window(
        self, buffer: deque, window_name: str
    ) -> Optional[BPMEstimate]:
        """
        Process a signal window and estimate BPM

        Args:
            buffer: Signal buffer
            window_name: Name for logging

        Returns:
            BPMEstimate or None
        """
        if len(buffer) < 30:
            return None

        signal_data = np.array(buffer)

        # Detrend
        detrended = self.detrend_signal(signal_data)

        # Bandpass filter
        filtered = self.apply_bandpass_filter(detrended)

        if filtered is None:
            return None

        # Estimate BPM
        estimation_method = self.processing_config.bpm.estimation_method

        if estimation_method == "welch":
            bpm, confidence = self.estimate_bpm_welch(filtered)
            method = "Welch"
        else:
            bpm, confidence = self.estimate_bpm_fft(filtered)
            method = "FFT"

        if bpm is None:
            return None

        # Validate BPM range
        bpm_config = self.processing_config.bpm
        if not (bpm_config.min <= bpm <= bpm_config.max):
            return None

        # Calculate quality
        quality = self.calculate_signal_quality(signal_data, filtered)

        return BPMEstimate(
            value=bpm,
            confidence=confidence,
            sqi=quality.sqi,
            snr=quality.snr,
            method=method,
        )

    def add_signal_value(self, value: float):
        """
        Add new signal value to buffers

        Args:
            value: Signal value to add
        """
        self.buffer_fast.append(value)
        self.buffer_slow.append(value)
        self.current_time += 1.0 / self.fps

    def should_update_fast(self) -> bool:
        """Check if fast window should be updated"""
        return (self.current_time - self.last_fast_update) >= self.fast_update_interval

    def should_update_slow(self) -> bool:
        """Check if slow window should be updated"""
        return (self.current_time - self.last_slow_update) >= self.slow_update_interval

    def update_estimates(self):
        """Update BPM estimates from buffers"""
        quality_config = self.processing_config.quality

        # Update fast window
        if self.should_update_fast():
            self.bpm_fast = self.process_signal_window(self.buffer_fast, "fast")
            self.last_fast_update = self.current_time

        # Update slow window
        if self.should_update_slow():
            self.bpm_slow = self.process_signal_window(self.buffer_slow, "slow")
            self.last_slow_update = self.current_time

        # Select active BPM based on quality
        if self.bpm_slow is not None and self.bpm_slow.sqi >= quality_config.sqi_high:
            # High quality slow estimate
            self.bpm_active = self.bpm_slow
        elif self.bpm_fast is not None and self.bpm_fast.sqi >= quality_config.sqi_low:
            # Acceptable fast estimate
            self.bpm_active = self.bpm_fast
        else:
            # Low quality, hold previous or clear
            if (
                self.last_valid_bpm is not None
                and self.motion_hold_time < self.max_hold_duration
            ):
                self.bpm_active = self.last_valid_bpm
                self.motion_hold_time += 1.0 / self.fps
            else:
                self.bpm_active = None

        # Apply EMA smoothing
        if self.bpm_active is not None:
            if self.bpm_ema is None:
                self.bpm_ema = self.bpm_active.value
            else:
                self.bpm_ema = (
                    self.ema_alpha * self.bpm_active.value
                    + (1 - self.ema_alpha) * self.bpm_ema
                )

            self.last_valid_bpm = self.bpm_active
            self.motion_hold_time = 0.0

    def get_current_bpm(self) -> Optional[float]:
        """Get current smoothed BPM estimate"""
        return self.bpm_ema

    def get_bpm_estimates(self) -> Dict[str, Optional[BPMEstimate]]:
        """Get all BPM estimates"""
        return {"fast": self.bpm_fast, "slow": self.bpm_slow, "active": self.bpm_active}

    def reset(self):
        """Reset all buffers and estimates"""
        self.buffer_fast.clear()
        self.buffer_slow.clear()
        self.bpm_fast = None
        self.bpm_slow = None
        self.bpm_active = None
        self.bpm_ema = None
        self.last_valid_bpm = None
        self.previous_peak_freq = None
        self.current_time = 0.0
        self.last_fast_update = 0.0
        self.last_slow_update = 0.0
        self.motion_hold_time = 0.0

        logger.info("SignalProcessor reset")

    def get_buffer_info(self) -> Dict[str, int]:
        """Get current buffer sizes"""
        fast_size = len(self.buffer_fast)
        slow_size = len(self.buffer_slow)
        active_buffer = self.bpm_active

        if active_buffer == self.bpm_fast:
            fill_percentage = (fast_size / self.fast_window_size) * 100
        else:
            fill_percentage = (slow_size / self.slow_window_size) * 100

        return {
            "fast": fast_size,
            "slow": slow_size,
            "fill_percentage": fill_percentage,
        }

    def get_spectrum_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get frequency spectrum data for visualization"""
        # Use active buffer
        if self.bpm_active == self.bpm_fast:
            signal_data = np.array(list(self.buffer_fast))
        else:
            signal_data = np.array(list(self.buffer_slow))

        # Need enough data for filtering and welch
        min_length = (
            max(
                len(self.b) if self.b is not None else 0,
                len(self.a) if self.a is not None else 0,
            )
            * 3
        )
        min_length = max(min_length, 60)  # At least 2 seconds of data

        if len(signal_data) < min_length:
            return None, None

        # Detrend and filter
        signal_detrended = self.detrend_signal(signal_data)
        signal_filtered = self.apply_bandpass_filter(signal_detrended)

        if signal_filtered is None or len(signal_filtered) < min_length:
            return None, None

        welch_config = self.processing_config.bpm.welch
        bpm_config = self.processing_config.bpm

        try:
            # Compute Welch periodogram
            nperseg = min(welch_config.nperseg, len(signal_filtered) // 2)
            noverlap = min(welch_config.noverlap, nperseg - 1)

            freqs, power = welch(
                signal_filtered,
                fs=self.fps,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="spectrum",
            )

            # Convert to BPM range
            freq_min = bpm_config.min / 60.0
            freq_max = bpm_config.max / 60.0

            # Filter to valid range
            valid_idx = (freqs >= freq_min) & (freqs <= freq_max)

            if not np.any(valid_idx):
                return None, None

            return freqs[valid_idx] * 60.0, power[valid_idx]  # Convert to BPM

        except Exception as e:
            logger.error(f"Spectrum computation failed: {e}")
            return None, None
