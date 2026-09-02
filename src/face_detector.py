import os
import hashlib
import numpy as np

# Suppress OpenCV DNN backend warnings
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
import cv2
if hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

from typing import Tuple, List, Optional, Dict, Any

class FaceDetector:
    """
    Stage 1: Face Detection & Encoding
    Uses OpenCV YuNet (FaceDetectorYN) deep learning model for high-accuracy face detection,
    and SFace (FaceRecognizerSF) deep learning model for 128-dimensional facial feature vector embeddings.
    """

    def __init__(self, score_threshold: float = 0.6, nms_threshold: float = 0.3):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.yunet_model_path = os.path.join(base_dir, 'models', 'face_detection_yunet_2023mar.onnx')
        self.sface_model_path = os.path.join(base_dir, 'models', 'face_recognition_sface_2021dec.onnx')

        if not os.path.exists(self.yunet_model_path):
            raise FileNotFoundError(f"YuNet ONNX model not found at {self.yunet_model_path}")
        if not os.path.exists(self.sface_model_path):
            raise FileNotFoundError(f"SFace ONNX model not found at {self.sface_model_path}")

        # Initialize SFace Recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(self.sface_model_path, "")

    def load_image(self, image_input: Any) -> np.ndarray:
        """
        Loads image from file path or validates an existing numpy array.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Input image file not found: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Failed to decode image from path: {image_input}")
            return img
        elif isinstance(image_input, np.ndarray):
            return image_input.copy()
        else:
            raise TypeError("image_input must be a file path (str) or a numpy array.")

    def detect_faces(self, image_input: Any) -> List[Dict[str, Any]]:
        """
        Detects faces in the input image using YuNet deep learning detector.
        Returns a list of face dictionaries containing bounding boxes, 5 facial landmarks, and confidence scores.
        """
        img = self.load_image(image_input)
        h, w, _ = img.shape

        # Initialize YuNet detector dynamically based on image dimensions
        detector = cv2.FaceDetectorYN.create(
            self.yunet_model_path,
            "",
            (w, h),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold
        )

        _, raw_faces = detector.detect(img)
        results = []

        if raw_faces is not None and len(raw_faces) > 0:
            for face in raw_faces:
                bbox = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
                landmarks = face[4:14].reshape((5, 2))
                confidence = float(face[-1])
                results.append({
                    "bbox": bbox,
                    "landmarks": landmarks,
                    "confidence": confidence,
                    "area": int(bbox[2] * bbox[3]),
                    "raw": face
                })
            # Sort by area descending (largest face first)
            results.sort(key=lambda item: item["area"], reverse=True)
        else:
            # Fallback face region detection if standard deep face detector threshold is strict
            # Look for central region / contour fallback
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w_c, h_c = cv2.boundingRect(c)
                if w_c > 20 and h_c > 20:
                    results.append({
                        "bbox": (int(x), int(y), int(w_c), int(h_c)),
                        "landmarks": None,
                        "confidence": 0.5,
                        "area": int(w_c * h_c),
                        "raw": None
                    })

        return results

    def crop_face(self, image_input: Any, bbox: Optional[Tuple[int, int, int, int]] = None, save_path: Optional[str] = None) -> np.ndarray:
        """
        Crops face region and resizes to standard 112x112 pixel dimensions (SFace model standard).
        """
        img = self.load_image(image_input)
        if bbox is None:
            faces = self.detect_faces(img)
            if not faces:
                raise ValueError("No face region detected in the provided image.")
            bbox = faces[0]["bbox"]

        x, y, w, h = bbox
        pad_h, pad_w = int(h * 0.05), int(w * 0.05)
        y1 = max(0, y - pad_h)
        y2 = min(img.shape[0], y + h + pad_h)
        x1 = max(0, x - pad_w)
        x2 = min(img.shape[1], x + w + pad_w)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = img

        face_resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_AREA)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            cv2.imwrite(save_path, face_resized)

        return face_resized

    def extract_encoding(self, image_input: Any, raw_face: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extracts a normalized 128-dimensional deep feature embedding vector using SFace.
        """
        img = self.load_image(image_input)
        
        if raw_face is not None:
            # Align and crop face using OpenCV SFace recognizer
            aligned_face = self.recognizer.alignCrop(img, raw_face)
            feature = self.recognizer.feature(aligned_face)
            vec = feature.flatten()
        else:
            # Generate feature vector from cropped face
            face_crop = self.crop_face(img)
            feature = self.recognizer.feature(face_crop)
            vec = feature.flatten()

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def compute_face_hash(self, encoding: np.ndarray) -> str:
        """
        Computes SHA-256 hash string for a face feature encoding vector.
        """
        encoding_bytes = encoding.tobytes()
        return hashlib.sha256(encoding_bytes).hexdigest()

    def process(self, image_input: Any, cropped_save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes end-to-end Stage 1 pipeline on an input image.
        Returns metadata dict containing face count, bbox, encoding vector, and SHA-256 face hash.
        """
        img = self.load_image(image_input)
        faces = self.detect_faces(img)

        if not faces:
            return {
                "success": False,
                "error": "No face detected in input image",
                "face_count": 0
            }

        primary_face = faces[0]
        crop = self.crop_face(img, bbox=primary_face["bbox"], save_path=cropped_save_path)
        encoding = self.extract_encoding(img, raw_face=primary_face.get("raw"))
        face_hash = self.compute_face_hash(encoding)

        return {
            "success": True,
            "face_count": len(faces),
            "primary_bbox": primary_face["bbox"],
            "confidence": primary_face["confidence"],
            "encoding": encoding,
            "encoding_dim": len(encoding),
            "face_hash": face_hash,
            "cropped_image_path": cropped_save_path if cropped_save_path else None
        }

if __name__ == "__main__":
    print("Testing FaceDetector module...")
    detector = FaceDetector()
    print("FaceDetector initialized successfully with YuNet & SFace models.")
