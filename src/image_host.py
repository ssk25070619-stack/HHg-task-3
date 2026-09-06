"""
Multi-Host Public Image Bridge for Reverse Search Engines
Uploads local cropped face images temporarily so external Google Lens / SerpAPI engines can analyze them.
Tries multiple hosts with graceful failover: Catbox, TmpFiles, Uguu, ImgBB.
"""

import os
import base64
from pathlib import Path
from typing import Optional
import requests

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def upload_to_tmpfiles(image_path: str) -> Optional[str]:
    """Uploads to tmpfiles.org and returns direct download link."""
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                "https://tmpfiles.org/api/v1/upload",
                files={"file": (os.path.basename(image_path), f, "image/jpeg")},
                headers=BROWSER_HEADERS,
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                raw_url = data.get("data", {}).get("url", "")
                if raw_url:
                    return raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/", 1)
    except Exception:
        pass
    return None

def upload_to_catbox(image_path: str) -> Optional[str]:
    """Uploads to catbox.moe and returns direct URL."""
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload"},
                files={"fileToUpload": f},
                headers=BROWSER_HEADERS,
                timeout=15
            )
            if resp.status_code == 200 and resp.text.strip().startswith("http"):
                return resp.text.strip()
    except Exception:
        pass
    return None

def upload_to_uguu(image_path: str) -> Optional[str]:
    """Uploads to uguu.se and returns direct URL."""
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                "https://uguu.se/upload",
                files={"files[]": (os.path.basename(image_path), f, "image/jpeg")},
                headers=BROWSER_HEADERS,
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                files = data.get("files", [])
                if files and files[0].get("url"):
                    return files[0]["url"].replace("\\/", "/")
    except Exception:
        pass
    return None

def upload_to_imgbb(image_path: str, api_key: Optional[str] = None) -> Optional[str]:
    """Uploads to imgbb.com if IMGBB_API_KEY is available."""
    key = api_key or os.getenv("IMGBB_API_KEY")
    if not key:
        return None
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        resp = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": key, "image": b64, "expiration": 600},
            timeout=15
        )
        if resp.status_code == 200:
            return resp.json().get("data", {}).get("url")
    except Exception:
        pass
    return None

def upload_public_image(image_path: str, imgbb_key: Optional[str] = None) -> Optional[str]:
    """
    Tries multiple providers in sequence until a public URL is obtained.
    """
    if not os.path.exists(image_path):
        return None

    # Try ImgBB if key exists
    url = upload_to_imgbb(image_path, imgbb_key)
    if url:
        return url

    # Try TmpFiles
    url = upload_to_tmpfiles(image_path)
    if url:
        return url

    # Try Catbox
    url = upload_to_catbox(image_path)
    if url:
        return url

    # Try Uguu
    url = upload_to_uguu(image_path)
    if url:
        return url

    return None
