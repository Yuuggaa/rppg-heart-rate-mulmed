"""
Modern Configuration System for rPPG Heart Rate Monitor
Type-safe configuration using dataclasses with YAML backend
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml


@dataclass
class ResolutionConfig:
    width: int = 1280
    height: int = 720


@dataclass
class ReconnectConfig:
    max_retries: int = 5
    backoff_intervals: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 30])


@dataclass
class HardwareConfig:
    camera_id: int = 0
    target_fps: int = 30
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    reconnect: ReconnectConfig = field(default_factory=ReconnectConfig)


@dataclass
class DNNModelConfig:
    prototxt: str = "models/deploy.prototxt"
    caffemodel: str = "models/res10_300x300_ssd_iter_140000.caffemodel"
    confidence: float = 0.6


@dataclass
class ROIGeometry:
    top_ratio: float
    bottom_ratio: float
    left_ratio: float
    right_ratio: float


@dataclass
class ROIRegion:
    name: str
    weight: float
    geometry: ROIGeometry


@dataclass
class SkinSegmentation:
    enabled: bool = True
    cb_range: Tuple[int, int] = (77, 127)
    cr_range: Tuple[int, int] = (133, 173)
    morph_kernel_size: int = 5


@dataclass
class ROIConfig:
    use_multi_region: bool = True
    regions: List[ROIRegion] = field(default_factory=list)
    skin_segmentation: SkinSegmentation = field(default_factory=SkinSegmentation)


@dataclass
class AlgorithmConfig:
    method: str = "chrom"
    face_detector: str = "auto"
    dnn_model: DNNModelConfig = field(default_factory=DNNModelConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)


@dataclass
class WindowConfig:
    duration: float
    update_interval: float


@dataclass
class WindowsConfig:
    fast: WindowConfig = field(default_factory=lambda: WindowConfig(10.0, 1.0))
    slow: WindowConfig = field(default_factory=lambda: WindowConfig(30.0, 5.0))


@dataclass
class BandpassConfig:
    low: float = 0.67
    high: float = 4.0


@dataclass
class FilterConfig:
    type: str = "cheby2"
    order: int = 5
    ripple_db: float = 40.0
    bandpass: BandpassConfig = field(default_factory=BandpassConfig)


@dataclass
class WelchConfig:
    nperseg: int = 256
    noverlap: int = 128


@dataclass
class PFTConfig:
    enabled: bool = True
    tracking_range: float = 0.3
    min_prominence: float = 0.1


@dataclass
class BPMConfig:
    min: int = 40
    max: int = 240
    estimation_method: str = "welch"
    welch: WelchConfig = field(default_factory=WelchConfig)
    pft: PFTConfig = field(default_factory=PFTConfig)


@dataclass
class SavGolConfig:
    window_length: int = 15
    polyorder: int = 3


@dataclass
class PolynomialConfig:
    order: int = 3


@dataclass
class DetrendingConfig:
    method: str = "savgol"
    savgol: SavGolConfig = field(default_factory=SavGolConfig)
    polynomial: PolynomialConfig = field(default_factory=PolynomialConfig)


@dataclass
class SmoothingConfig:
    enabled: bool = True
    ema_alpha: float = 0.3


@dataclass
class CrossValidationConfig:
    enabled: bool = True
    tolerance: int = 5


@dataclass
class QualityConfig:
    sqi_high: float = 0.75
    sqi_low: float = 0.2
    snr_min: float = 0.0
    cross_validation: CrossValidationConfig = field(
        default_factory=CrossValidationConfig
    )


@dataclass
class ProcessingConfig:
    windows: WindowsConfig = field(default_factory=WindowsConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    bpm: BPMConfig = field(default_factory=BPMConfig)
    detrending: DetrendingConfig = field(default_factory=DetrendingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)


@dataclass
class LKFeatureParams:
    max_corners: int = 100
    quality_level: float = 0.3
    min_distance: int = 7
    block_size: int = 7


@dataclass
class SparseLKConfig:
    feature_params: LKFeatureParams = field(default_factory=LKFeatureParams)
    threshold: float = 1.5


@dataclass
class DenseFarnebackConfig:
    threshold: float = 2.0


@dataclass
class VarianceConfig:
    enabled: bool = True
    threshold: float = 0.15
    buffer_size: int = 60


@dataclass
class MotionConfig:
    method: str = "sparse_lk"
    sparse_lk: SparseLKConfig = field(default_factory=SparseLKConfig)
    dense_farneback: DenseFarnebackConfig = field(default_factory=DenseFarnebackConfig)
    variance: VarianceConfig = field(default_factory=VarianceConfig)
    grace_period: float = 3.0
    hold_duration: float = 10.0
    no_face_timeout: float = 5.0


@dataclass
class GammaConfig:
    enabled: bool = True
    auto_adjust: bool = True
    default_value: float = 1.2
    range: Tuple[float, float] = (0.8, 2.0)


@dataclass
class CLAHEConfig:
    enabled: bool = True
    clip_limit: float = 2.0
    tile_size: Tuple[int, int] = (8, 8)


@dataclass
class ChangeDetectionConfig:
    enabled: bool = True
    threshold: int = 20
    history_size: int = 30
    grace_period: float = 2.0


@dataclass
class LightingConfig:
    gamma: GammaConfig = field(default_factory=GammaConfig)
    clahe: CLAHEConfig = field(default_factory=CLAHEConfig)
    change_detection: ChangeDetectionConfig = field(
        default_factory=ChangeDetectionConfig
    )


@dataclass
class WindowDisplayConfig:
    title: str = "rPPG Heart Rate Monitor"
    width: int = 1600
    height: int = 900
    fullscreen: bool = False


@dataclass
class ThemeConfig:
    style: str = "glassmorphism"
    background_color: Tuple[int, int, int] = (10, 14, 39)
    primary_color: Tuple[int, int, int] = (0, 255, 255)
    secondary_color: Tuple[int, int, int] = (0, 255, 127)
    warning_color: Tuple[int, int, int] = (255, 165, 0)
    error_color: Tuple[int, int, int] = (255, 69, 0)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    alpha: float = 0.7


@dataclass
class ROIColors:
    forehead: Tuple[int, int, int] = (255, 0, 0)
    left_cheek: Tuple[int, int, int] = (0, 255, 0)
    right_cheek: Tuple[int, int, int] = (0, 0, 255)


@dataclass
class VideoFeedPanel:
    enabled: bool = True
    show_roi: bool = True
    show_landmarks: bool = False
    roi_colors: ROIColors = field(default_factory=ROIColors)


@dataclass
class SignalWaveformPanel:
    enabled: bool = True
    duration: float = 10.0
    show_raw: bool = True
    show_filtered: bool = True
    normalize: bool = True


@dataclass
class FrequencySpectrumPanel:
    enabled: bool = True
    show_harmonics: bool = True
    freq_range: Tuple[float, float] = (0.5, 4.5)


@dataclass
class BPMHistoryPanel:
    enabled: bool = True
    duration: float = 60.0
    show_fast: bool = True
    show_slow: bool = True


@dataclass
class StatusIndicatorsPanel:
    enabled: bool = True
    show_fps: bool = True
    show_sqi: bool = True
    show_snr: bool = True
    show_warnings: bool = True


@dataclass
class PanelsConfig:
    video_feed: VideoFeedPanel = field(default_factory=VideoFeedPanel)
    signal_waveform: SignalWaveformPanel = field(default_factory=SignalWaveformPanel)
    frequency_spectrum: FrequencySpectrumPanel = field(
        default_factory=FrequencySpectrumPanel
    )
    bpm_history: BPMHistoryPanel = field(default_factory=BPMHistoryPanel)
    status_indicators: StatusIndicatorsPanel = field(
        default_factory=StatusIndicatorsPanel
    )


@dataclass
class DebugConfig:
    enabled: bool = False
    show_latency: bool = True
    show_buffer_size: bool = True
    show_frame_drops: bool = True


@dataclass
class DisplayConfig:
    window: WindowDisplayConfig = field(default_factory=WindowDisplayConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    panels: PanelsConfig = field(default_factory=PanelsConfig)
    update_rate: int = 30
    debug: DebugConfig = field(default_factory=DebugConfig)


@dataclass
class AsyncTasksConfig:
    capture_priority: str = "high"
    process_priority: str = "normal"
    display_priority: str = "normal"


@dataclass
class QueuesConfig:
    capture_to_process: int = 5
    process_to_display: int = 3


@dataclass
class OptimizationConfig:
    auto_optimize: bool = True
    adaptive_quality: bool = True
    skip_frames: bool = False
    max_latency_ms: int = 100


@dataclass
class MonitoringConfig:
    enabled: bool = True
    check_interval: float = 5.0
    log_performance: bool = True


@dataclass
class PerformanceConfig:
    async_tasks: AsyncTasksConfig = field(default_factory=AsyncTasksConfig)
    queues: QueuesConfig = field(default_factory=QueuesConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


@dataclass
class FileLoggingConfig:
    enabled: bool = True
    directory: str = "logs"
    max_size_mb: int = 5
    backup_count: int = 3
    rotation: str = "size"


@dataclass
class ConsoleLoggingConfig:
    enabled: bool = True
    colorized: bool = True


@dataclass
class ExportConfig:
    signal_data: bool = False
    csv_output: str = "data/signal_export.csv"
    include_timestamps: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: FileLoggingConfig = field(default_factory=FileLoggingConfig)
    console: ConsoleLoggingConfig = field(default_factory=ConsoleLoggingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


@dataclass
class AdaptationConfig:
    enabled: bool = False
    range: float = 0.2


@dataclass
class CalibrationConfig:
    auto_calibrate: bool = False
    duration: float = 10.0
    save_results: bool = True
    calibration_file: str = "user_calibration.yaml"
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)


@dataclass
class Config:
    """Main configuration container"""

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file"""
        path = Path(yaml_path)
        if not path.exists():
            # Return default config if file doesn't exist
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Recursively convert dict to Config dataclass"""
        import typing

        def convert_value(value, target_type):
            # Handle Optional types
            if (
                hasattr(target_type, "__origin__")
                and target_type.__origin__ is typing.Union
            ):
                # Get the non-None type from Optional
                args = [arg for arg in target_type.__args__ if arg is not type(None)]
                if args:
                    target_type = args[0]

            if isinstance(value, dict):
                if hasattr(target_type, "__dataclass_fields__"):
                    # It's a dataclass
                    return target_type(
                        **{
                            k: convert_value(
                                v, target_type.__dataclass_fields__[k].type
                            )
                            for k, v in value.items()
                            if k in target_type.__dataclass_fields__
                        }
                    )
                return value
            elif isinstance(value, list):
                # Handle List types
                if (
                    hasattr(target_type, "__origin__")
                    and target_type.__origin__ is list
                ):
                    if hasattr(target_type, "__args__") and target_type.__args__:
                        item_type = target_type.__args__[0]
                        return [convert_value(item, item_type) for item in value]
                return value
            return value

        kwargs = {}
        for field_name, field_info in cls.__dataclass_fields__.items():
            if field_name in data:
                kwargs[field_name] = convert_value(data[field_name], field_info.type)

        return cls(**kwargs)

    def to_yaml(self, yaml_path: str = "config.yaml"):
        """Save configuration to YAML file"""
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._to_dict(), f, default_flow_style=False, sort_keys=False)

    def _to_dict(self) -> dict:
        """Convert dataclass to dict recursively"""

        def convert_value(value):
            if hasattr(value, "__dataclass_fields__"):
                return {
                    k: convert_value(getattr(value, k))
                    for k in value.__dataclass_fields__
                }
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            return value

        return {
            field_name: convert_value(getattr(self, field_name))
            for field_name in self.__dataclass_fields__
        }


# Global configuration instance
_config: Optional[Config] = None


def load_config(yaml_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file and cache it"""
    global _config
    _config = Config.from_yaml(yaml_path)
    return _config


def get_config() -> Config:
    """Get cached configuration instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
