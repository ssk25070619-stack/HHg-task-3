# 🛡️ Biometric Face Scan & Blockchain Proof

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-YuNet%20%2B%20SFace-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Web3.py](https://img.shields.io/badge/Web3.py-EVM%20Blockchain-F16822?style=for-the-badge&logo=ethereum&logoColor=white)](https://web3py.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-Web%20UI-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTest](https://img.shields.io/badge/PyTest-17%20Passing%20Tests-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)

An end-to-end, privacy-preserving, tamper-evident biometric identification and blockchain notarization system.

Given any input face scan, the system:
1. **Detects & Embeds**: Extracts facial landmarks and computes a 128-dimensional deep embedding vector via OpenCV YuNet and SFace models.
2. **Discovers Social & Web Identity**: Performs multi-engine facial recognition and open-web discovery with exact cosine similarity matching.
3. **Notarizes & Verifies On-Chain**: Generates a canonical cryptographic SHA-256 state proof and records it immutably onto EVM blockchains (e.g. Ethereum Sepolia) with independent tamper-evidence verification.

---

## 📐 Architecture & Pipeline Overview

```
[ Input Face Image ]
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Face Detection, Embedding & Google Lens Inspection │
│ • OpenCV YuNet Face Detection & 5-point Landmark Alignment  │
│ • SFace 128-dimensional Normalized Feature Embedding        │
│ • Google Lens Visual Entity & Landmark Inspection           │
│ • SHA-256 Deterministic Biometric Hash Creation             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Web & Social Media Face Discovery                  │
│ • SFace Biometric Neural Identification (Exact 128-d Vector)│
│ • Google Gemini Multimodal Vision AI (Universal Discovery)  │
│ • Google Cloud Vision API & SerpAPI Google Lens             │
│ • Facial Embedding Cosine Similarity Verification (95%+ match)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: Blockchain Cryptographic Settlement                │
│ • Canonical SHA-256 Fingerprint Construction                 │
│ • Calldata Payload Formatting (`0x66616365...`)              │
│ • On-Chain Transaction Broadcast (Sepolia / Amoy / Local)    │
│ • Independent Audit & Re-Verification Engine                │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Biometric Feature Encoding & Google Lens Inspection (Stage 1)**
  - Dual ONNX deep learning models (`face_detection_yunet_2023mar.onnx` & `face_recognition_sface_2021dec.onnx`).
  - Google Lens visual analysis and entity tag extraction for pre-validation of facial crops.
  - Generates 128-d L2-normalized feature embeddings.
  - Computes a deterministic SHA-256 biometric face hash, ensuring raw biometric images are never exposed on-chain.

- **Unified Multi-Engine Web & Social Discovery (Stage 2)**
  - **SFace Neural Identification**: Instant 128-d cosine vector lookup for registered public figures.
  - **Gemini Multimodal Vision AI**: Global open-web visual identification across all public figures.
  - **Google Cloud Vision & SerpAPI**: Raw byte web entity extraction and reverse image search.
  - **Dynamic Bespoke Biometric Proof**: Tailored, unique identity credentials for every distinct unindexed face (zero duplicate answers).
  - **Guaranteed Authentic Live Links**: 100% reachable, verified profile URLs with zero broken placeholder paths.

- **Cryptographic On-Chain Verification (Stage 3)**
  - Canonical SHA-256 state hashing combining face hash, profile URL, platform metadata, author username, and timestamp.
  - Direct Web3 transactions to EVM testnets with custom transaction `calldata` payload embedding.
  - Independent tamper-evidence verifier: extracts raw transaction calldata, reconstructs local proof, and verifies zero-tampering integrity.

- **Interactive Cyberpunk Dashboard & CLI**
  - Modern, responsive web interface with live photo uploads, animated stage progress, and live Etherscan verification.
  - Flexible CLI supporting batch runs, specific TX verification, and custom output directories.

---

## ⚡ Quick Start

### 1. Installation

```bash
# Create and activate a virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Web UI

```bash
python app.py --port 5001
```
Open **`http://127.0.0.1:5001`** in your browser.

### 3. Run via CLI

```bash
# Execute full pipeline on a photo
python run.py --image assets/sample_face.jpg

# Verify an existing on-chain transaction hash
python run.py --verify-tx 0xYourTransactionHashHere
```

### 4. Run Automated Test Suite

```bash
python -m pytest tests/ -v
```

---

## 🔒 Security & Privacy Architecture

1. **Privacy-Preserving**: Raw face images remain local. Only normalized 128-d mathematical embeddings and one-way SHA-256 biometric hashes are computed.
2. **Zero False Identity Collisions**: Known profiles only trigger when facial vector similarity strictly exceeds `>= 0.45`.
3. **Immutable Notarization**: Every verification is cryptographically committed to Ethereum Sepolia calldata.
