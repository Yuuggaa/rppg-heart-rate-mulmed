"""
Modern Face Detection System with Dual Backend Support
Supports MediaPipe Face Mesh and OpenCV DNN with automatic fallback
Uses adaptive ROI extraction based on facial geometry ratios
"""

import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FaceLandmarks:
    """Container for face detection results"""

    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None


class FaceDetectorBackend(ABC):
    """Abstract base class for face detection backends"""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        """Detect face in frame"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get backend name"""
        pass


class MediaPipeFaceDetector(FaceDetectorBackend):
    """MediaPipe-based face detector with landmark extraction"""

    def __init__(self):
        self.available = False
        try:
            import mediapipe as mp

            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.available = True
            logger.info("MediaPipe Face Detector initialized successfully")
        except Exception as e:
            logger.warning(f"MediaPipe not available: {e}")
            self.face_mesh = None

    def detect(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        if not self.available or self.face_mesh is None:
            return None

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]

                # Extract landmark coordinates
                landmarks = []
                xs, ys = [], []

                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))
                    xs.append(x)
                    ys.append(y)

                # Compute bounding box from landmarks
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

                return FaceLandmarks(bbox=bbox, confidence=1.0, landmarks=landmarks)

            return None

        except Exception as e:
            logger.error(f"MediaPipe detection error: {e}")
            return None

    def is_available(self) -> bool:
        return self.available

    def get_name(self) -> str:
        return "MediaPipe"


class DNNFaceDetector(FaceDetectorBackend):
    """OpenCV DNN-based face detector using Caffe model"""

    def __init__(self, config):
        self.available = False
        self.net = None
        self.confidence_threshold = config.algorithm.dnn_model.confidence

        try:
            # Try to load DNN model
            prototxt = config.algorithm.dnn_model.prototxt
            caffemodel = config.algorithm.dnn_model.caffemodel

            prototxt_path = Path(prototxt)
            caffemodel_path = Path(caffemodel)

            if prototxt_path.exists() and caffemodel_path.exists():
                self.net = cv2.dnn.readNetFromCaffe(
                    str(prototxt_path), str(caffemodel_path)
                )
                self.available = True
                logger.info("OpenCV DNN Face Detector initialized successfully")
            else:
                logger.warning(f"DNN model files not found: {prototxt}, {caffemodel}")

        except Exception as e:
            logger.warning(f"DNN detector not available: {e}")

    def detect(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        if not self.available or self.net is None:
            return None

        try:
            h, w = frame.shape[:2]

            # Prepare blob
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )

            # Forward pass
            self.net.setInput(blob)
            detections = self.net.forward()

            # Find best detection
            best_detection = None
            best_confidence = 0.0

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if (
                    confidence > self.confidence_threshold
                    and confidence > best_confidence
                ):
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)

                    # Ensure valid bounding box
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    if x2 > x1 and y2 > y1:
                        best_detection = (x1, y1, x2 - x1, y2 - y1)
                        best_confidence = confidence

            if best_detection is not None:
                return FaceLandmarks(
                    bbox=best_detection,
                    confidence=float(best_confidence),
                    landmarks=None,
                )

            return None

        except Exception as e:
            logger.error(f"DNN detection error: {e}")
            return None

    def is_available(self) -> bool:
        return self.available

    def get_name(self) -> str:
        return "OpenCV DNN"


class HaarCascadeFaceDetector(FaceDetectorBackend):
    """Haar Cascade fallback face detector (always available)"""

    def __init__(self):
        self.available = False
        self.cascade = None

        try:
            # Use OpenCV's built-in Haar Cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)

            if not self.cascade.empty():
                self.available = True
                logger.info("Haar Cascade Face Detector initialized (fallback mode)")
            else:
                logger.warning("Haar Cascade failed to load")

        except Exception as e:
            logger.warning(f"Haar Cascade not available: {e}")

    def detect(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        """Detect face using Haar Cascade"""
        if not self.available:
            return None

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )

            if len(faces) == 0:
                return None

            # Take largest face
            areas = [w * h for (x, y, w, h) in faces]
            largest_idx = np.argmax(areas)
            x, y, w, h = faces[largest_idx]

            return FaceLandmarks(
                bbox=(x, y, w, h),
                confidence=0.8,  # Fixed confidence for Haar
                landmarks=None,
            )

        except Exception as e:
            logger.error(f"Haar Cascade detection error: {e}")
            return None

    def is_available(self) -> bool:
        return self.available

    def get_name(self) -> str:
        return "Haar Cascade"


class AdaptiveROIExtractor:
    """
    Adaptive ROI extraction using facial geometry ratios
    Does not rely on hardcoded landmark indices
    """

    def __init__(self, config):
        self.config = config
        self.roi_configs = {
            region.name: region for region in config.algorithm.roi.regions
        }
        self.skin_config = config.algorithm.roi.skin_segmentation

    def extract_rois(
        self, frame: np.ndarray, face_result: FaceLandmarks
    ) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Extract multiple ROIs from face based on geometry ratios

        Args:
            frame: Input frame
            face_result: Face detection result

        Returns:
            Dictionary of roi_name -> (roi_image, skin_mask)
        """
        rois = {}

        x, y, w, h = face_result.bbox

        for roi_name, roi_config in self.roi_configs.items():
            geometry = roi_config.geometry

            # Calculate ROI bounds using ratios
            roi_x1 = x + int(w * geometry.left_ratio)
            roi_x2 = x + int(w * geometry.right_ratio)
            roi_y1 = y + int(h * geometry.top_ratio)
            roi_y2 = y + int(h * geometry.bottom_ratio)

            # Clip to frame bounds
            roi_x1 = max(0, roi_x1)
            roi_x2 = min(frame.shape[1], roi_x2)
            roi_y1 = max(0, roi_y1)
            roi_y2 = min(frame.shape[0], roi_y2)

            # Extract ROI
            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()

                # Apply skin segmentation if enabled
                skin_mask = None
                if self.skin_config.enabled:
                    skin_mask = self._segment_skin(roi_image)

                rois[roi_name] = (roi_image, skin_mask)

        return rois

    def _segment_skin(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Segment skin pixels using YCbCr color space

        Args:
            roi: Input ROI image

        Returns:
            Binary skin mask or None
        """
        if roi is None or roi.size == 0:
            return None

        try:
            # Convert to YCbCr
            ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

            # Extract Cb and Cr channels
            cb = ycrcb[:, :, 1]
            cr = ycrcb[:, :, 2]

            # Apply thresholds
            cb_min, cb_max = self.skin_config.cb_range
            cr_min, cr_max = self.skin_config.cr_range

            skin_mask = (
                (cb >= cb_min) & (cb <= cb_max) & (cr >= cr_min) & (cr <= cr_max)
            ).astype(np.uint8) * 255

            # Morphological operations
            kernel_size = self.skin_config.morph_kernel_size
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )

            # Close then open
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

            return skin_mask

        except Exception as e:
            logger.error(f"Skin segmentation error: {e}")
            return None

    def get_roi_bboxes(
        self, face_bbox: Tuple[int, int, int, int]
    ) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Get bounding boxes for all ROIs without extracting images
        Used for visualization

        Args:
            face_bbox: Face bounding box (x, y, w, h)

        Returns:
            Dictionary of roi_name -> (x, y, w, h)
        """
        x, y, w, h = face_bbox
        bboxes = {}

        for roi_name, roi_config in self.roi_configs.items():
            geometry = roi_config.geometry

            roi_x1 = x + int(w * geometry.left_ratio)
            roi_x2 = x + int(w * geometry.right_ratio)
            roi_y1 = y + int(h * geometry.top_ratio)
            roi_y2 = y + int(h * geometry.bottom_ratio)

            roi_w = roi_x2 - roi_x1
            roi_h = roi_y2 - roi_y1

            bboxes[roi_name] = (roi_x1, roi_y1, roi_w, roi_h)

        return bboxes


class FaceDetectionSystem:
    """
    Main face detection system with dual backend support and automatic fallback
    """

    def __init__(self, config):
        self.config = config
        self.roi_extractor = AdaptiveROIExtractor(config)

        # Initialize backends
        self.backends: List[FaceDetectorBackend] = []

        detection_mode = config.algorithm.face_detector

        if detection_mode in ["auto", "mediapipe"]:
            mp_backend = MediaPipeFaceDetector()
            if mp_backend.is_available():
                self.backends.append(mp_backend)

        if detection_mode in ["auto", "dnn"]:
            dnn_backend = DNNFaceDetector(config)
            if dnn_backend.is_available():
                self.backends.append(dnn_backend)

        # Always add Haar Cascade as fallback
        if detection_mode == "auto":
            haar_backend = HaarCascadeFaceDetector()
            if haar_backend.is_available():
                self.backends.append(haar_backend)

        if not self.backends:
            raise RuntimeError("No face detection backend available!")

        self.current_backend_idx = 0
        self.no_face_counter = 0

        backend_names = [b.get_name() for b in self.backends]
        logger.info(f"Face Detection System initialized with backends: {backend_names}")
        logger.info(
            f"Active backend: {self.backends[self.current_backend_idx].get_name()}"
        )

    def detect_face(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        """
        Detect face using current backend with automatic fallback

        Args:
            frame: Input BGR frame

        Returns:
            FaceLandmarks or None
        """
        # Try current backend
        result = self.backends[self.current_backend_idx].detect(frame)

        if result is not None:
            self.no_face_counter = 0
            return result

        # Try fallback backends
        for i, backend in enumerate(self.backends):
            if i == self.current_backend_idx:
                continue

            result = backend.detect(frame)
            if result is not None:
                logger.info(f"Switched to fallback backend: {backend.get_name()}")
                self.current_backend_idx = i
                self.no_face_counter = 0
                return result

        self.no_face_counter += 1
        return None

    def extract_rois(
        self, frame: np.ndarray
    ) -> Tuple[
        Optional[FaceLandmarks], Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]
    ]:
        """
        Detect face and extract ROIs

        Args:
            frame: Input BGR frame

        Returns:
            Tuple of (face_result, rois_dict)
        """
        face_result = self.detect_face(frame)

        if face_result is None:
            return None, {}

        rois = self.roi_extractor.extract_rois(frame, face_result)

        return face_result, rois

    def get_roi_bboxes_for_visualization(
        self, face_result: FaceLandmarks
    ) -> Dict[str, Tuple[int, int, int, int]]:
        """Get ROI bounding boxes for drawing on frame"""
        return self.roi_extractor.get_roi_bboxes(face_result.bbox)

    def get_no_face_counter(self) -> int:
        """Get number of consecutive frames with no face"""
        return self.no_face_counter

    def get_current_backend(self) -> str:
        """Get name of current active backend"""
        return self.backends[self.current_backend_idx].get_name()

    def release(self):
        """Release resources"""
        logger.info("FaceDetectionSystem released")
