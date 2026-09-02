import pytest
import os
from src.blockchain_verifier import BlockchainVerifier

@pytest.fixture
def blockchain_verifier():
    return BlockchainVerifier()

@pytest.fixture
def sample_payload():
    return {
        "schema": "HHGoa-Task3-BiometricProof-v1.0",
        "face_hash": "0cea2a5ba9435c9fe1d54ff06b51aa367f6ede1f89139b606e5a9d68529cc6b5",
        "matched_platform": "Twitter / X",
        "post_url": "https://x.com/dev_innovator/status/1765893489123_sample",
        "post_title": "Autonomous Agent & AI Developer Community Update",
        "post_author": "@dev_innovator",
        "confidence_score": 87.5,
        "timestamp": "2026-08-31T14:32:00Z"
    }

def test_blockchain_verifier_initialization(blockchain_verifier):
    assert blockchain_verifier is not None
    assert blockchain_verifier.explorer_base is not None

def test_compute_payload_hash_determinism(blockchain_verifier, sample_payload):
    hash1 = blockchain_verifier.compute_payload_hash(sample_payload)
    hash2 = blockchain_verifier.compute_payload_hash(sample_payload)
    
    assert isinstance(hash1, str)
    assert len(hash1) == 64
    assert hash1 == hash2

def test_compute_payload_hash_mutation_sensitivity(blockchain_verifier, sample_payload):
    hash_original = blockchain_verifier.compute_payload_hash(sample_payload)
    
    mutated = dict(sample_payload)
    mutated["post_author"] = "@attacker_spoof"
    hash_mutated = blockchain_verifier.compute_payload_hash(mutated)
    
    assert hash_original != hash_mutated

def test_upload_and_authentic_verification(blockchain_verifier, sample_payload):
    metadata = {
        "primary_post": {
            "platform": sample_payload["matched_platform"],
            "url": sample_payload["post_url"],
            "title": sample_payload["post_title"],
            "author": sample_payload["post_author"],
            "confidence_score": sample_payload["score"] if "score" in sample_payload else 87.5,
            "timestamp": sample_payload["timestamp"]
        }
    }
    
    upload_res = blockchain_verifier.upload_record(sample_payload["raw_face"] if "raw_face" in sample_payload else sample_payload["face_hash"], metadata)
    
    assert upload_res["success"] is True
    assert upload_res["tx_hash"].startswith("0x")
    assert len(upload_res["tx_hash"]) == 66
    assert upload_res["block_number"] > 0
    assert upload_res["payload_hash"] is not None
    
    # Re-verify original data
    stored_record = upload_res["stored_record"]
    verify_res = blockchain_verifier.verify_record(stored_record, upload_res["tx_hash"])
    
    assert verify_res["success"] is True
    assert verify_res["is_verified"] is True
    assert verify_res["tamper_detected"] is False
    assert verify_res["status"] == "AUTHENTIC"

def test_tamper_detection_on_mutated_data(blockchain_verifier, sample_payload):
    metadata = {
        "primary_post": {
            "platform": sample_payload["matched_platform"],
            "url": sample_payload["post_url"],
            "title": sample_payload["post_title"],
            "author": sample_payload["post_author"],
            "confidence_score": 87.5,
            "timestamp": sample_payload["timestamp"]
        }
    }
    
    upload_res = blockchain_verifier.upload_record(sample_payload["face_hash"], metadata)
    tx_hash = upload_res["tx_hash"]
    
    # Tamper with the payload (modify URL)
    tampered_record = dict(upload_res["stored_record"])
    tampered_record["post_url"] = "https://fake-phishing-link.com"
    
    verify_res = blockchain_verifier.verify_record(tampered_record, tx_hash)
    
    assert verify_res["success"] is True
    assert verify_res["is_verified"] is False
    assert verify_res["tamper_detected"] is True
    assert verify_res["status"] == "TAMPER_DETECTED"
    assert verify_res["computed_hash"] != verify_res["on_chain_hash"]

def test_verify_nonexistent_transaction(blockchain_verifier, sample_payload):
    fake_tx = "0x9999999999999999999999999999999999999999999999999999999999999999"
    res = blockchain_verifier.verify_record(sample_payload, fake_tx)
    assert res["success"] is False
    assert res["is_verified"] is False
    assert res["tamper_detected"] is True
