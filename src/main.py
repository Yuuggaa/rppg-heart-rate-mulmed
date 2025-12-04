"""
Async Main Application
Orchestrates async tasks for rPPG heart rate monitoring
"""

import asyncio
import logging
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from utils import setup_logging, PerformanceMonitor, AsyncRateLimiter
from camera_handler import AsyncCameraHandler
from face_detector import FaceDetectionSystem
from chrom_extractor import CHROMExtractor
from signal_processor import SignalProcessor
from lighting_handler import LightingHandler
from motion_detector import MotionDetector
from visualizer import OpenCVVisualizer

logger = logging.getLogger(__name__)


class AsyncRPPGSystem:
    """Main async rPPG system orchestrator"""

    def __init__(self):
        # Load configuration
        self.config = load_config()

        # Setup logging
        setup_logging(self.config)

        # Initialize components
        self.camera: AsyncCameraHandler = None
        self.face_detector: FaceDetectionSystem = None
        self.chrom_extractor: CHROMExtractor = None
        self.signal_processor: SignalProcessor = None
        self.lighting_handler: LightingHandler = None
        self.motion_detector: MotionDetector = None
        self.visualizer: OpenCVVisualizer = None

        # Async queues
        self.frame_queue: asyncio.Queue = None
        self.result_queue: asyncio.Queue = None

        # Control
        self.running = False
        self.tasks = []

        # Performance
        self.performance_monitor = PerformanceMonitor(self.config)

        logger.info("AsyncRPPGSystem initialized")

    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing system components...")

        # Camera
        self.camera = AsyncCameraHandler(self.config)
        await self.camera.initialize()

        fps = self.camera.get_fps()

        # Face detection
        self.face_detector = FaceDetectionSystem(self.config)

        # CHROM extractor
        self.chrom_extractor = CHROMExtractor(self.config)

        # Signal processor
        self.signal_processor = SignalProcessor(self.config, int(fps))

        # Lighting handler
        self.lighting_handler = LightingHandler(self.config)

        # Motion detector
        self.motion_detector = MotionDetector(self.config)

        # Visualizer
        self.visualizer = OpenCVVisualizer(self.config)

        # Queues
        queue_config = self.config.performance.queues
        self.frame_queue = asyncio.Queue(maxsize=queue_config.capture_to_process)
        self.result_queue = asyncio.Queue(maxsize=queue_config.process_to_display)

        logger.info("All components initialized successfully")

    async def capture_loop(self):
        """Async capture loop - high priority"""
        logger.info("Capture loop started")

        rate_limiter = AsyncRateLimiter(self.camera.get_fps())
        frame_count = 0

        try:
            while self.running:
                # Rate limiting
                await rate_limiter.acquire()

                # Read frame
                ret, frame = await self.camera.read_frame()

                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    await asyncio.sleep(0.1)
                    continue

                # Process lighting
                processed_frame, lighting_stable = self.lighting_handler.process_frame(
                    frame
                )

                # Put in queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(
                        {
                            "frame": processed_frame,
                            "original": frame,
                            "lighting_stable": lighting_stable,
                            "timestamp": asyncio.get_event_loop().time(),
                        }
                    )
                except asyncio.QueueFull:
                    # Drop frame if queue full
                    pass

                frame_count += 1

                # Update performance
                if frame_count % 30 == 0:
                    await self.performance_monitor.update_metrics(
                        "capture", self.camera.get_fps()
                    )

        except asyncio.CancelledError:
            logger.info("Capture loop cancelled")
        except Exception as e:
            logger.error(f"Capture loop error: {e}", exc_info=True)

    async def process_loop(self):
        """Async processing loop"""
        logger.info("Process loop started")

        process_count = 0

        try:
            while self.running:
                # Get frame from queue
                try:
                    frame_data = await asyncio.wait_for(
                        self.frame_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                frame = frame_data["frame"]
                lighting_stable = frame_data["lighting_stable"]

                # Face detection and ROI extraction
                face_result, rois = self.face_detector.extract_rois(frame)

                if face_result is None:
                    # No face detected
                    no_face_count = self.face_detector.get_no_face_counter()

                    if no_face_count > 150:  # ~5 seconds at 30 FPS
                        # Reset buffers
                        self.chrom_extractor.reset()
                        self.signal_processor.reset()
                        self.motion_detector.reset()

                    # Send status to display
                    try:
                        self.result_queue.put_nowait(
                            {
                                "type": "no_face",
                                "frame": frame,
                                "message": "No face detected",
                            }
                        )
                    except asyncio.QueueFull:
                        pass

                    continue

                # Motion detection
                motion_magnitude, motion_detected_flow = (
                    self.motion_detector.detect_motion_optical_flow(
                        frame, face_result.bbox
                    )
                )

                # Log motion magnitude every 300 frames (10 seconds)
                if process_count % 300 == 0:
                    logger.debug(f"Motion magnitude: {motion_magnitude:.2f} pixels")

                # Extract CHROM signal
                signal_value = self.chrom_extractor.process_frame_rois(rois)

                if signal_value is None:
                    logger.warning("CHROM signal is None - ROI extraction failed")

                # Variance-based motion detection
                motion_detected_var = False
                if signal_value is not None:
                    motion_detected_var = self.motion_detector.detect_motion_variance(
                        signal_value
                    )

                # Update motion state
                motion_active = self.motion_detector.update_motion_state(
                    motion_detected_flow, motion_detected_var
                )

                # Process signal if no motion and lighting stable
                if signal_value is not None and not motion_active and lighting_stable:
                    self.signal_processor.add_signal_value(signal_value)
                    self.signal_processor.update_estimates()

                # Get BPM estimate
                current_bpm = self.signal_processor.get_current_bpm()
                bpm_estimates = self.signal_processor.get_bpm_estimates()

                # Get spectrum data
                spectrum_freqs, spectrum_power = (
                    self.signal_processor.get_spectrum_data()
                )

                # Prepare result
                result = {
                    "type": "success",
                    "frame": frame,
                    "face_bbox": face_result.bbox,
                    "rois": self.face_detector.get_roi_bboxes_for_visualization(
                        face_result
                    ),
                    "bpm": current_bpm,
                    "bpm_estimates": bpm_estimates,
                    "motion_detected": motion_active,
                    "motion_magnitude": motion_magnitude,
                    "lighting_stable": lighting_stable,
                    "signal_value": signal_value,
                    "backend": self.face_detector.get_current_backend(),
                    "buffer_info": self.signal_processor.get_buffer_info(),
                    "spectrum_freqs": spectrum_freqs,
                    "spectrum_power": spectrum_power,
                }

                # Send to display
                try:
                    self.result_queue.put_nowait(result)
                except asyncio.QueueFull:
                    pass

                process_count += 1

                # Small yield to prevent blocking
                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            logger.info("Process loop cancelled")
        except Exception as e:
            logger.error(f"Process loop error: {e}", exc_info=True)

    async def display_loop(self):
        """Async display loop"""
        logger.info("Display loop started")

        rate_limiter = AsyncRateLimiter(self.config.display.update_rate)

        try:
            while self.running:
                await rate_limiter.acquire()

                # Get result from queue
                try:
                    result = self.result_queue.get_nowait()
                except asyncio.QueueEmpty:
                    # Show empty frame or last frame
                    await asyncio.sleep(0.01)
                    continue

                # Update visualizer
                if result["type"] == "success":
                    self.visualizer.update(result)
                elif result["type"] == "no_face":
                    self.visualizer.show_message(result["frame"], result["message"])

                # Render frame
                display_frame = self.visualizer.render()

                # Show frame
                cv2.imshow(self.config.display.window.title, display_frame)

                # Check for key press
                key = cv2.waitKey(1) & 0xFF

                if key == 27 or key == ord("q"):  # ESC or Q
                    logger.info("User requested exit")
                    self.running = False
                    break
                elif key == ord("r"):  # Reset
                    self.chrom_extractor.reset()
                    self.signal_processor.reset()
                    self.motion_detector.reset()
                    logger.info("System reset by user")
                elif key == ord("d"):  # Toggle debug
                    self.visualizer.toggle_debug()

        except asyncio.CancelledError:
            logger.info("Display loop cancelled")
        except Exception as e:
            logger.error(f"Display loop error: {e}", exc_info=True)
        finally:
            cv2.destroyAllWindows()

    async def run(self):
        """Run the async system"""
        logger.info("Starting rPPG system...")

        await self.initialize()

        self.running = True

        # Create tasks
        self.tasks = [
            asyncio.create_task(self.capture_loop(), name="capture"),
            asyncio.create_task(self.process_loop(), name="process"),
            asyncio.create_task(self.display_loop(), name="display"),
        ]

        # Wait for all tasks
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("System cancelled")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown system gracefully"""
        logger.info("Shutting down system...")

        self.running = False

        # Cancel tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for cancellation
        await asyncio.gather(*self.tasks, return_exceptions=True)

        # Release camera
        if self.camera:
            await self.camera.release()

        # Close visualizer
        cv2.destroyAllWindows()

        logger.info("System shutdown complete")


async def main_async():
    """Main async entry point"""
    system = AsyncRPPGSystem()

    try:
        await system.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
    finally:
        await system.shutdown()


if __name__ == "__main__":
    # Windows-specific event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main_async())
