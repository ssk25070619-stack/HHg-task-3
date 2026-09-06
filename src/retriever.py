"""
Secure Candidate Image Retriever
Downloads candidate images from external URLs with SSRF protection, size limits, and magic byte validation.
"""

import io
import ipaddress
import urllib.parse
from typing import Optional, Tuple
import requests

MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024  # 10 MB limit
TIMEOUT_SECONDS = 5

MAGIC_BYTES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp"
}

def is_safe_url(url: str) -> bool:
    """
    Validates URL scheme and protects against SSRF attacks on localhost and private networks.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return False

        hostname = parsed.hostname or ""
        if not hostname or hostname.lower() in ["localhost", "127.0.0.1", "::1", "0.0.0.0"]:
            return False

        # Check private IP ranges
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except ValueError:
            pass

        return True
    except Exception:
        return False

def safe_download_image(url: str) -> Optional[bytes]:
    """
    Downloads image bytes safely with stream limiting and format validation.
    """
    if not is_safe_url(url):
        return None

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        with requests.get(url, headers=headers, timeout=TIMEOUT_SECONDS, stream=True) as resp:
            if resp.status_code != 200:
                return None

            content = bytearray()
            for chunk in resp.iter_content(chunk_size=8192):
                content.extend(chunk)
                if len(content) > MAX_DOWNLOAD_BYTES:
                    return None  # Exceeded max size

            # Validate magic bytes
            valid = False
            for magic in MAGIC_BYTES:
                if bytes(content).startswith(magic) or (b"WEBP" in bytes(content[:16])):
                    valid = True
                    break

            return bytes(content) if valid else None
    except Exception:
        return None
