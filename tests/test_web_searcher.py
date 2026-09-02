import pytest
import os
import numpy as np
from src.web_searcher import WebSearcher
from src.face_detector import FaceDetector

@pytest.fixture
def web_searcher():
    return WebSearcher()

@pytest.fixture
def face_detector():
    return FaceDetector()

def test_web_searcher_initialization(web_searcher):
    assert web_searcher is not None

def test_identify_platform(web_searcher):
    assert web_searcher.identify_platform('https://twitter.com/user/status/123') == 'Twitter / X'
    assert web_searcher.identify_platform('https://x.com/user/status/123') == 'Twitter / X'
    assert web_searcher.identify_platform('https://www.linkedin.com/in/alex') == 'LinkedIn'
    assert web_searcher.identify_platform('https://github.com/torvalds') == 'GitHub'
    assert web_searcher.identify_platform('https://unknown-domain.org/article') == 'Web Article'

def test_cosine_similarity_identical_and_orthogonal(web_searcher):
    vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Identical vectors should yield cosine similarity of 1.0
    sim_identical = web_searcher.compute_cosine_similarity(vec1, vec2)
    assert pytest.approx(sim_identical, 0.001) == 1.0

    # Orthogonal vectors should yield cosine similarity of 0.0
    sim_ortho = web_searcher.compute_cosine_similarity(vec1, vec3)
    assert pytest.approx(sim_ortho, 0.001) == 0.0

def test_similarity_to_confidence_scaling(web_searcher):
    conf_high = web_searcher.similarity_to_confidence(1.0)
    conf_mid = web_searcher.similarity_to_confidence(0.5)
    conf_low = web_searcher.similarity_to_confidence(-0.2)

    assert conf_high == 100.0
    assert 0.0 <= conf_mid <= 100.0
    assert conf_low == 0.0

def test_search_by_face_nonexistent_image(web_searcher):
    res = web_searcher.search_by_face('non_existent_path.jpg')
    assert res['success'] is False
    assert res['status'] == 'error'

def test_search_by_face_with_sample_image(web_searcher, face_detector):
    sample_img = 'assets/sample_face.jpg'
    assert os.path.exists(sample_img)

    stage1 = face_detector.process(sample_img)
    assert stage1['success'] is True

    res = web_searcher.search_by_face(
        image_path=sample_img,
        face_hash=stage1['face_hash'],
        face_encoding=stage1['encoding'],
        face_detector=face_detector
    )

    assert res['success'] is True
    assert res['total_found'] >= 1
    assert 'primary_match' in res
    assert res['primary_match'] is not None
    assert 'url' in res['primary_match']
    assert 'platform' in res['primary_match']
    assert 'confidence_score' in res['primary_match']
    assert res['primary_match']['confidence_score'] > 0
