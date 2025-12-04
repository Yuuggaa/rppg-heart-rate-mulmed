"""
Modern Utility Functions
Helper functions for async rPPG system
"""

import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)


def setup_logging(config) -> str:
    """
    Setup structured logging system
    
    Args:
        config: Configuration object
        
    Returns:
        Path to log file
    """
    logging_config = config.logging
    
    # Create logs directory
    log_dir = Path(logging_config.file.directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Session log file
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"rppg_session_{session_time}.log"
    
    # Get log level
    log_level = getattr(logging, logging_config.level.upper(), logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler
    if logging_config.file.enabled:
        max_bytes = logging_config.file.max_size_mb * 1024 * 1024
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=logging_config.file.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
    
    # Console handler
    console_formatter = logging.Formatter(
        "%(levelname)-8s | %(name)-20s | %(message)s"
    )
    console_handler = logging.StreamHandler()
    
    if logging_config.console.enabled:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.WARNING)
    
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    
    if logging_config.file.enabled:
        root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log startup
    logging.info("=" * 80)
    logging.info(f"rPPG System Started - Session: {session_time}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Log level: {logging_config.level}")
    logging.info("=" * 80)
    
    return str(log_file)


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio
    
    Args:
        signal: Signal data
        noise: Noise data
        
    Returns:
        SNR in dB
    """
    signal_power = np.var(signal)
    noise_power = np.var(noise)
    
    if noise_power < 1e-10:
        return 0.0
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def calculate_sqi(signal: np.ndarray, snr: float, variance: float, periodicity: float) -> float:
    """
    Calculate Signal Quality Index
    
    Args:
        signal: Signal data
        snr: Signal-to-noise ratio
        variance: Variance metric
        periodicity: Periodicity metric
        
    Returns:
        SQI value (0-1)
    """
    # Normalize SNR (0-20 dB range)
    snr_normalized = np.clip(snr / 20.0, 0, 1)
    
    # Weighted combination
    sqi = (0.4 * snr_normalized + 
           0.3 * variance + 
           0.3 * periodicity)
    
    return np.clip(sqi, 0, 1)


class PerformanceMonitor:
    """Monitor async task performance"""
    
    def __init__(self, config):
        self.config = config
        self.monitoring_config = config.performance.monitoring
        
        self.enabled = self.monitoring_config.enabled
        self.check_interval = self.monitoring_config.check_interval
        
        self.metrics = {
            'capture_fps': 0.0,
            'process_fps': 0.0,
            'display_fps': 0.0,
            'latency_ms': 0.0,
            'queue_sizes': {}
        }
        
        self.last_check = 0.0
    
    async def update_metrics(self, task_name: str, fps: float, latency: Optional[float] = None):
        """Update performance metrics"""
        if not self.enabled:
            return
        
        self.metrics[f'{task_name}_fps'] = fps
        
        if latency is not None:
            self.metrics['latency_ms'] = latency
    
    async def check_performance(self):
        """Check if performance is acceptable"""
        if not self.enabled:
            return
        
        current_time = asyncio.get_event_loop().time()
        
        if current_time - self.last_check < self.check_interval:
            return
        
        self.last_check = current_time
        
        # Log metrics
        if self.monitoring_config.log_performance:
            logger.info(f"Performance: Capture {self.metrics['capture_fps']:.1f} FPS, "
                       f"Process {self.metrics['process_fps']:.1f} FPS, "
                       f"Display {self.metrics['display_fps']:.1f} FPS, "
                       f"Latency {self.metrics['latency_ms']:.1f} ms")
    
    def get_metrics(self) -> dict:
        """Get current metrics"""
        return self.metrics.copy()


class AsyncRateLimiter:
    """Async rate limiter for controlling task frequency"""
    
    def __init__(self, rate: float):
        """
        Initialize rate limiter
        
        Args:
            rate: Target rate in Hz
        """
        self.interval = 1.0 / rate if rate > 0 else 0
        self.last_time = 0.0
    
    async def acquire(self):
        """Wait until next interval"""
        if self.interval <= 0:
            return
        
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - self.last_time
        
        if elapsed < self.interval:
            await asyncio.sleep(self.interval - elapsed)
        
        self.last_time = asyncio.get_event_loop().time()


def validate_bpm_range(bpm: float, config) -> bool:
    """
    Validate if BPM is in physiological range
    
    Args:
        bpm: BPM value to validate
        config: Configuration object
        
    Returns:
        True if valid
    """
    bpm_config = config.processing.bpm
    return bpm_config.min <= bpm <= bpm_config.max


def smooth_signal(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth signal using moving average
    
    Args:
        signal: Input signal
        window_size: Window size for smoothing
        
    Returns:
        Smoothed signal
    """
    if len(signal) < window_size:
        return signal
    
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(signal, kernel, mode='same')
    
    return smoothed
