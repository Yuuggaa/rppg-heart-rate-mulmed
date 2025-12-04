"""
OpenCV Native Visualizer
Custom visualization using OpenCV instead of matplotlib
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OpenCVWidget:
    """Base widget for custom OpenCV GUI"""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Render widget on canvas"""
        raise NotImplementedError


class SignalPlotWidget(OpenCVWidget):
    """Real-time signal plot widget"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        max_samples: int = 300,
        title: str = "Signal",
    ):
        super().__init__(x, y, width, height)
        self.max_samples = max_samples
        self.title = title
        self.data = deque(maxlen=max_samples)

        # Colors
        self.bg_color = (30, 30, 30)
        self.grid_color = (50, 50, 50)
        self.line_color = (0, 255, 128)
        self.text_color = (200, 200, 200)

    def add_value(self, value: float):
        """Add new value to plot"""
        self.data.append(value)

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Render signal plot"""
        # Background with glassmorphism
        roi = canvas[self.y : self.y + self.height, self.x : self.x + self.width]
        overlay = np.full_like(roi, self.bg_color, dtype=np.uint8)
        cv2.addWeighted(overlay, 0.7, roi, 0.3, 0, roi)

        # Border
        cv2.rectangle(
            canvas,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            (80, 80, 80),
            2,
        )

        # Title
        cv2.putText(
            canvas,
            self.title,
            (self.x + 10, self.y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.text_color,
            1,
        )

        if len(self.data) < 2:
            return canvas

        # Calculate plot area
        plot_x = self.x + 10
        plot_y = self.y + 30
        plot_w = self.width - 20
        plot_h = self.height - 40

        # Draw grid
        for i in range(5):
            y_grid = plot_y + int(i * plot_h / 4)
            cv2.line(
                canvas, (plot_x, y_grid), (plot_x + plot_w, y_grid), self.grid_color, 1
            )

        # Normalize data
        data_array = np.array(self.data)
        if np.ptp(data_array) > 0:
            data_norm = (data_array - np.min(data_array)) / np.ptp(data_array)
        else:
            data_norm = np.zeros_like(data_array)

        # Draw signal line
        points = []
        for i, val in enumerate(data_norm):
            x = plot_x + int(i * plot_w / self.max_samples)
            y = plot_y + plot_h - int(val * plot_h)
            points.append((x, y))

        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i + 1], self.line_color, 2)

        return canvas


class FrequencyPlotWidget(OpenCVWidget):
    """Frequency spectrum plot widget"""

    def __init__(
        self, x: int, y: int, width: int, height: int, title: str = "Spectrum"
    ):
        super().__init__(x, y, width, height)
        self.title = title
        self.freqs = None
        self.power = None

        # Colors
        self.bg_color = (30, 30, 30)
        self.bar_color = (255, 128, 0)
        self.peak_color = (0, 255, 255)
        self.text_color = (200, 200, 200)

    def update_spectrum(self, freqs: np.ndarray, power: np.ndarray):
        """Update spectrum data"""
        self.freqs = freqs
        self.power = power

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Render frequency spectrum"""
        # Background
        roi = canvas[self.y : self.y + self.height, self.x : self.x + self.width]
        overlay = np.full_like(roi, self.bg_color, dtype=np.uint8)
        cv2.addWeighted(overlay, 0.7, roi, 0.3, 0, roi)

        # Border
        cv2.rectangle(
            canvas,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            (80, 80, 80),
            2,
        )

        # Title
        cv2.putText(
            canvas,
            self.title,
            (self.x + 10, self.y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.text_color,
            1,
        )

        if self.freqs is None or self.power is None or len(self.power) == 0:
            # Show "Collecting data..."
            cv2.putText(
                canvas,
                "Collecting data...",
                (self.x + self.width // 2 - 60, self.y + self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 100, 100),
                1,
            )
            return canvas
            cv2.putText(
                canvas,
                "Collecting data...",
                (self.x + self.width // 2 - 60, self.y + self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 100, 100),
                1,
            )
            return canvas

        # Plot area
        plot_x = self.x + 10
        plot_y = self.y + 30
        plot_w = self.width - 20
        plot_h = self.height - 50

        # Normalize power
        power_norm = (
            self.power / np.max(self.power) if np.max(self.power) > 0 else self.power
        )

        # Draw bars
        num_bars = min(len(power_norm), 50)
        bar_width = max(plot_w // num_bars - 2, 1)

        peak_idx = np.argmax(power_norm)

        for i in range(num_bars):
            idx = int(i * len(power_norm) / num_bars)
            bar_h = int(power_norm[idx] * plot_h)
            bar_x = plot_x + int(i * plot_w / num_bars)
            bar_y = plot_y + plot_h - bar_h

            color = self.peak_color if idx == peak_idx else self.bar_color
            cv2.rectangle(
                canvas, (bar_x, bar_y), (bar_x + bar_width, plot_y + plot_h), color, -1
            )

        # Frequency labels
        cv2.putText(
            canvas,
            f"{self.freqs[0]:.1f} Hz",
            (plot_x, plot_y + plot_h + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            self.text_color,
            1,
        )
        cv2.putText(
            canvas,
            f"{self.freqs[-1]:.1f} Hz",
            (plot_x + plot_w - 30, plot_y + plot_h + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            self.text_color,
            1,
        )

        return canvas


class BPMHistoryWidget(OpenCVWidget):
    """BPM history line plot widget"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        max_samples: int = 100,
        title: str = "BPM History",
    ):
        super().__init__(x, y, width, height)
        self.max_samples = max_samples
        self.title = title
        self.bpm_history = deque(maxlen=max_samples)

        # Colors
        self.bg_color = (30, 30, 30)
        self.grid_color = (50, 50, 50)
        self.line_color = (0, 128, 255)
        self.text_color = (200, 200, 200)

        # BPM range
        self.min_bpm = 40
        self.max_bpm = 180

    def add_bpm(self, bpm: Optional[float]):
        """Add BPM value"""
        if bpm is not None:
            self.bpm_history.append(bpm)

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Render BPM history"""
        # Background
        roi = canvas[self.y : self.y + self.height, self.x : self.x + self.width]
        overlay = np.full_like(roi, self.bg_color, dtype=np.uint8)
        cv2.addWeighted(overlay, 0.7, roi, 0.3, 0, roi)

        # Border
        cv2.rectangle(
            canvas,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            (80, 80, 80),
            2,
        )

        # Title
        cv2.putText(
            canvas,
            self.title,
            (self.x + 10, self.y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.text_color,
            1,
        )

        if len(self.bpm_history) < 2:
            # Show "Collecting data..."
            cv2.putText(
                canvas,
                "Collecting data...",
                (self.x + self.width // 2 - 60, self.y + self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 100, 100),
                1,
            )
            return canvas

        # Plot area
        plot_x = self.x + 40
        plot_y = self.y + 30
        plot_w = self.width - 50
        plot_h = self.height - 40

        # Draw grid and labels
        for i, bpm_val in enumerate([60, 90, 120, 150]):
            if self.min_bpm <= bpm_val <= self.max_bpm:
                y_pos = (
                    plot_y
                    + plot_h
                    - int(
                        (bpm_val - self.min_bpm)
                        / (self.max_bpm - self.min_bpm)
                        * plot_h
                    )
                )
                cv2.line(
                    canvas,
                    (plot_x, y_pos),
                    (plot_x + plot_w, y_pos),
                    self.grid_color,
                    1,
                )
                cv2.putText(
                    canvas,
                    str(bpm_val),
                    (self.x + 5, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    self.text_color,
                    1,
                )

        # Draw BPM line
        points = []
        for i, bpm in enumerate(self.bpm_history):
            if self.min_bpm <= bpm <= self.max_bpm:
                x = plot_x + int(i * plot_w / self.max_samples)
                y = (
                    plot_y
                    + plot_h
                    - int((bpm - self.min_bpm) / (self.max_bpm - self.min_bpm) * plot_h)
                )
                points.append((x, y))

        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i + 1], self.line_color, 2)

        return canvas


class StatusPanelWidget(OpenCVWidget):
    """Status information panel"""

    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        self.status_items = {}

        # Colors
        self.bg_color = (30, 30, 30)
        self.text_color = (200, 200, 200)
        self.label_color = (150, 150, 150)
        self.value_color = (100, 255, 100)
        self.warning_color = (0, 165, 255)

    def update_status(self, key: str, value: str, is_warning: bool = False):
        """Update status item"""
        self.status_items[key] = (value, is_warning)

    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Render status panel"""
        # Background
        roi = canvas[self.y : self.y + self.height, self.x : self.x + self.width]
        overlay = np.full_like(roi, self.bg_color, dtype=np.uint8)
        cv2.addWeighted(overlay, 0.7, roi, 0.3, 0, roi)

        # Border
        cv2.rectangle(
            canvas,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            (80, 80, 80),
            2,
        )

        # Render items
        y_offset = 20
        for key, (value, is_warning) in self.status_items.items():
            # Label
            cv2.putText(
                canvas,
                f"{key}:",
                (self.x + 10, self.y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.label_color,
                1,
            )

            # Value
            value_color = self.warning_color if is_warning else self.value_color
            cv2.putText(
                canvas,
                value,
                (self.x + 120, self.y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                value_color,
                1,
            )

            y_offset += 20

        return canvas


class OpenCVVisualizer:
    """Main OpenCV visualizer"""

    def __init__(self, config):
        self.config = config

        # Display config
        self.width = config.display.window.width
        self.height = config.display.window.height

        # Create widgets
        self._create_widgets()

        # State
        self.debug_mode = False
        self.last_result = None

        logger.info("OpenCVVisualizer initialized")

    def _create_widgets(self):
        """Create UI widgets"""
        # Video feed area (left side)
        self.video_width = 640
        self.video_height = 480

        # Right panel width
        panel_width = self.width - self.video_width - 30

        # Create widgets
        widget_height = 150
        spacing = 10

        self.signal_plot = SignalPlotWidget(
            self.video_width + 20, 10, panel_width, widget_height, title="CHROM Signal"
        )

        self.spectrum_plot = FrequencyPlotWidget(
            self.video_width + 20,
            widget_height + spacing + 10,
            panel_width,
            widget_height,
            title="Frequency Spectrum",
        )

        self.bpm_history = BPMHistoryWidget(
            self.video_width + 20,
            2 * widget_height + 2 * spacing + 10,
            panel_width,
            widget_height,
            title="BPM History",
        )

        self.status_panel = StatusPanelWidget(
            self.video_width + 20,
            3 * widget_height + 3 * spacing + 10,
            panel_width,
            self.height - 3 * widget_height - 3 * spacing - 20,
        )

    def update(self, result: Dict):
        """Update visualizer with new result"""
        self.last_result = result

        # Update signal plot
        if result.get("signal_value") is not None:
            self.signal_plot.add_value(result["signal_value"])

        # Update spectrum plot
        spectrum_freqs = result.get("spectrum_freqs")
        spectrum_power = result.get("spectrum_power")
        if spectrum_freqs is not None and spectrum_power is not None:
            logger.debug(f"Updating spectrum: {len(spectrum_freqs)} points")
            self.spectrum_plot.update_spectrum(spectrum_freqs, spectrum_power)

        # Update BPM history
        bpm = result.get("bpm")
        if bpm is not None:
            logger.debug(f"Adding BPM to history: {bpm:.1f}")
            self.bpm_history.add_bpm(bpm)

        # Update status
        self.status_panel.update_status(
            "BPM", f"{result.get('bpm', 0):.1f}" if result.get("bpm") else "---"
        )
        self.status_panel.update_status("Backend", result.get("backend", "N/A"))
        self.status_panel.update_status(
            "Motion",
            "DETECTED" if result.get("motion_detected") else "Stable",
            is_warning=result.get("motion_detected", False),
        )
        self.status_panel.update_status(
            "Lighting",
            "Stable" if result.get("lighting_stable") else "UNSTABLE",
            is_warning=not result.get("lighting_stable", True),
        )

        buffer_info = result.get("buffer_info", {})
        self.status_panel.update_status(
            "Buffer", f"{buffer_info.get('fill_percentage', 0):.0f}%"
        )

    def show_message(self, frame: np.ndarray, message: str):
        """Show message on frame"""
        self.last_result = {"type": "message", "frame": frame, "message": message}

    def toggle_debug(self):
        """Toggle debug mode"""
        self.debug_mode = not self.debug_mode
        logger.info(f"Debug mode: {self.debug_mode}")

    def render(self) -> np.ndarray:
        """Render complete visualization"""
        # Create canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if self.last_result is None:
            return canvas

        # Resize frame
        frame = self.last_result.get("frame")
        if frame is not None:
            frame_resized = cv2.resize(frame, (self.video_width, self.video_height))

            # Draw face bbox and ROIs if available
            if self.last_result.get("type") == "success":
                face_bbox = self.last_result.get("face_bbox")
                if face_bbox:
                    scale_x = self.video_width / frame.shape[1]
                    scale_y = self.video_height / frame.shape[0]

                    x, y, w, h = face_bbox
                    x_scaled = int(x * scale_x)
                    y_scaled = int(y * scale_y)
                    w_scaled = int(w * scale_x)
                    h_scaled = int(h * scale_y)

                    # Draw face box
                    cv2.rectangle(
                        frame_resized,
                        (x_scaled, y_scaled),
                        (x_scaled + w_scaled, y_scaled + h_scaled),
                        (0, 255, 0),
                        2,
                    )

                    # Draw ROIs
                    rois = self.last_result.get("rois", {})
                    colors = [(255, 0, 0), (0, 255, 255), (255, 128, 0)]
                    for i, (roi_name, roi_bbox) in enumerate(rois.items()):
                        rx, ry, rw, rh = roi_bbox
                        rx_s = int(rx * scale_x)
                        ry_s = int(ry * scale_y)
                        rw_s = int(rw * scale_x)
                        rh_s = int(rh * scale_y)
                        cv2.rectangle(
                            frame_resized,
                            (rx_s, ry_s),
                            (rx_s + rw_s, ry_s + rh_s),
                            colors[i % len(colors)],
                            1,
                        )

            # Place frame on canvas
            canvas[0 : self.video_height, 0 : self.video_width] = frame_resized

        # Render widgets
        self.signal_plot.render(canvas)
        self.spectrum_plot.render(canvas)
        self.bpm_history.render(canvas)
        self.status_panel.render(canvas)

        # Draw BPM display (large)
        if self.last_result.get("bpm"):
            bpm_text = f"{self.last_result['bpm']:.1f}"
            text_size = cv2.getTextSize(bpm_text, cv2.FONT_HERSHEY_DUPLEX, 2, 3)[0]
            text_x = (self.video_width - text_size[0]) // 2
            text_y = self.video_height - 30

            # Shadow
            cv2.putText(
                canvas,
                bpm_text,
                (text_x + 2, text_y + 2),
                cv2.FONT_HERSHEY_DUPLEX,
                2,
                (0, 0, 0),
                3,
            )
            # Text
            cv2.putText(
                canvas,
                bpm_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                2,
                (0, 255, 128),
                3,
            )
            # BPM label
            cv2.putText(
                canvas,
                "BPM",
                (text_x + text_size[0] + 10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

        # Message overlay
        if self.last_result.get("type") == "message":
            message = self.last_result.get("message", "")
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (self.video_width - text_size[0]) // 2
            text_y = self.video_height // 2

            cv2.putText(
                canvas,
                message,
                (text_x + 2, text_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                canvas,
                message,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 165, 255),
                2,
            )

        # Debug info
        if self.debug_mode and self.last_result.get("type") == "success":
            debug_text = [
                f"Motion Mag: {self.last_result.get('motion_magnitude', 0):.2f}",
                f"Signal: {self.last_result.get('signal_value', 0):.4f}",
            ]
            y_offset = 20
            for text in debug_text:
                cv2.putText(
                    canvas,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1,
                )
                y_offset += 15

        # Controls hint
        hint_text = "ESC/Q: Quit | R: Reset | D: Debug"
        cv2.putText(
            canvas,
            hint_text,
            (10, self.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
        )

        return canvas
