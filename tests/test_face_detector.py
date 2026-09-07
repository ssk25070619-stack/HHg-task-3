import pytest
import os
import numpy as np
import cv2
from src.face_detector import FaceDetector

@pytest.fixture
def face_detector():
    return FaceDetector()

@pytest.fixture
def sample_image_path(tmp_path):
    img_path = str(tmp_path / "test_face.jpg")
    img = np.full((300, 300, 3), (240, 240, 240), dtype=np.uint8)
    # Draw simple facial geometry
    cv2.circle(img, (150, 150), 70, (200, 170, 140), -1)
    cv2.circle(img, (125, 130), 10, (50, 50, 50), -1)
    cv2.circle(img, (175, 130), 10, (50, 50, 50), -1)
    cv2.ellipse(img, (150, 175), (30, 15), 0, 0, 180, (50, 50, 150), 4)
    cv2.imwrite(img_path, img)
    return img_path

def test_face_detector_initialization(face_detector):
    assert face_detector is not None
    assert face_detector.recognizer is not None

def test_load_image_invalid_path(face_detector):
    with pytest.raises(FileNotFoundError):
        face_detector.load_image("non_existent_file.jpg")

def test_extract_encoding_shape_and_norm(face_detector, sample_image_path):
    encoding = face_detector.extract_encoding(sample_image_path)
    
    assert isinstance(encoding, np.ndarray)
    assert len(encoding) == 128
    norm = np.linalg.norm(encoding)
    assert pytest.approx(norm, 0.01) == 1.0

def test_compute_face_hash(face_detector):
    dummy_vec = np.ones(128, dtype=np.float32)
    hash1 = face_detector.compute_face_hash(dummy_vec)
    hash2 = face_detector.compute_face_hash(dummy_vec)
    
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 hex string length
    assert hash1 == hash2

def test_detect_with_google_lens(face_detector, sample_image_path):
    lens_res = face_detector.detect_with_google_lens(sample_image_path)
    assert lens_res is not None
    assert lens_res.get("lens_detected") is True
    assert "engine" in lens_res
    assert "detected_label" in lens_res
    assert "visual_tags" in lens_res

def test_process_pipeline(face_detector, sample_image_path, tmp_path):
    crop_out = str(tmp_path / "crop.jpg")
    result = face_detector.process(sample_image_path, cropped_save_path=crop_out)
    
    assert result["success"] is True
    assert result["face_count"] >= 1
    assert "face_hash" in result
    assert len(result["face_hash"]) == 64
    assert len(result["encoding"]) == 128
    assert os.path.exists(crop_out)
    assert "google_lens" in result
    assert result["google_lens"]["lens_detected"] is True
