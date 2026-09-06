"""
Canonical Evidence Manifest & Cryptographic Hasher
Generates a deterministic SHA-256 evidence fingerprint for blockchain anchoring.
Privacy rule: Raw face image bytes and high-dimensional embeddings are excluded.
"""

import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

def canonical_json_dumps(data: Any) -> str:
    """
    Serializes a dict/list to deterministic canonical JSON (sorted keys, compact separators).
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def create_evidence_manifest(
    face_hash: str,
    primary_post: Dict[str, Any],
    face_bbox: Optional[List[int]] = None,
    total_matches: int = 1,
    pipeline_version: str = "3.0.0"
) -> Dict[str, Any]:
    """
    Creates a privacy-preserving canonical evidence manifest.
    """
    sim = primary_post.get("cosine_similarity", 0.90)
    conf = primary_post.get("confidence_score", 90.0)

    manifest = {
        "pipeline_version": pipeline_version,
        "face_hash": face_hash,
        "face_bbox": face_bbox or [0, 0, 0, 0],
        "match": {
            "platform": primary_post.get("platform", "Instagram"),
            "author": primary_post.get("author", "@creator"),
            "url": primary_post.get("url", ""),
            "title": primary_post.get("title", ""),
            "cosine_similarity": f"{float(sim):.6f}",
            "confidence_score": f"{float(conf):.2f}",
            "verified": bool(primary_post.get("match_verified", True))
        },
        "search_engine": primary_post.get("search_engine", "Multi-Engine Vision & Social Cascade"),
        "total_matches": int(total_matches),
        "timestamp": primary_post.get("timestamp") or datetime.now(timezone.utc).isoformat()
    }
    return manifest

def compute_evidence_fingerprint(manifest: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 fingerprint from the canonical manifest.
    """
    canon = canonical_json_dumps(manifest)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()
