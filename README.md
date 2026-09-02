# 🛡️ HH Goa 2026: Face Identification & Blockchain Verification Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-YuNet%20%2B%20SFace-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Web3.py](https://img.shields.io/badge/Web3.py-EVM%20Blockchain-F16822?style=for-the-badge&logo=ethereum&logoColor=white)](https://web3py.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-Web%20UI-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTest](https://img.shields.io/badge/PyTest-17%20Passing%20Tests-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)

An end-to-end, privacy-preserving, tamper-evident verification pipeline developed for **HH Goa 2026**.

Given a face scan image, the system:
1. Detects facial features and computes a 128-dimensional deep embedding vector via OpenCV YuNet and SFace.
2. Performs real-time reverse image discovery across social media platforms (Twitter/X, LinkedIn, GitHub, Instagram, Reddit) with biometric cosine similarity matching.
3. Generates a canonical cryptographic SHA-256 state proof and anchors it immutably onto EVM blockchains (Sepolia / Polygon Amoy / Local EVM node) with independent re-verification capabilities.

---

## 📐 Architecture & Pipeline Overview

```
[ Input Face Image ]
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Face Detection & Biometric Embedding               │
│ • OpenCV YuNet Face Detection & Crop                        │
│ • SFace 128-dimensional Normalized Feature Vector          │
│ • SHA-256 Deterministic Biometric Hash Creation             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Web & Social Media Reverse Image Search            │
│ • SerpAPI (Google Lens) Reverse Image Discovery             │
│ • Social Platform Profile Scraping & Identification         │
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

- **Biometric Feature Encoding (Stage 1)**
  - Dual ONNX deep learning models (`face_detection_yunet_2023mar.onnx` & `face_recognition_sface_2021dec.onnx`).
  - Generates 128-d L2-normalized feature embeddings.
  - Computes a deterministic SHA-256 biometric face hash, ensuring raw images are never exposed on-chain.

- **Web & Social Media Discovery (Stage 2)**
  - Integrates **SerpAPI Google Lens** reverse image search for live web matches.
  - Automatically identifies matching profiles on **Twitter/X, LinkedIn, GitHub, Instagram, Reddit, YouTube, and Facebook**.
  - Calculates biometric cosine similarity between query face and profile image targets.
  - Includes a resilient **offline zero-dependency fallback** engine for development without API rate limits.

- **Cryptographic On-Chain Verification (Stage 3)**
  - Canonical SHA-256 state hashing combining face hash, post URL, platform metadata, author username, and ISO timestamp.
  - Direct Web3 transactions to EVM testnets with custom transaction `calldata` payload embedding.
  - Independent tamper-evidence verifier: extracts raw transaction calldata, reconstructs local proof, and verifies zero-tampering integrity.

- **Dual Interfaces (CLI + Interactive Web Dashboard)**
  - Modern, responsive **Flask Web UI** with drag-and-drop file uploader, interactive confidence metrics, and on-chain verification viewer.
  - Flexible **CLI Entry Point** supporting batch runs, specific TX verification, and custom output directories.

- **Automated Test Suite**
  - **17 PyTest unit and integration tests** covering face detection, web search, similarity math, blockchain calldata construction, and Flask Web UI routes.

---

## 📁 Repository Structure

```
HHGoa-Task3/
├── app.py                      # Flask Web Dashboard backend & API endpoints
├── run.py                      # Main CLI entry point & launcher
├── requirements.txt            # Python dependency manifest
├── .env.example                # Environment variable configuration template
├── assets/                     # Sample input face images
│   ├── sample_face.jpg
│   └── test.jpg
├── src/                        # Core pipeline packages
│   ├── face_detector.py        # Stage 1: YuNet detection & SFace embedding
│   ├── web_searcher.py         # Stage 2: SerpAPI social search & cosine similarity
│   ├── blockchain_verifier.py  # Stage 3: Web3 transaction & on-chain verification
│   └── pipeline.py             # End-to-end master controller
├── templates/                  # Flask HTML UI template
│   └── index.html
├── static/                     # Web UI CSS/JS assets
│   ├── css/style.css
│   └── js/app.js
└── tests/                      # Automated PyTest suite
    ├── test_face_detector.py
    ├── test_web_searcher.py
    ├── test_blockchain_verifier.py
    └── test_pipeline.py
```

---

## ⚡ Quick Start

### 1. Prerequisites & Installation

Ensure Python 3.9+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/devesh1905/HHgoaTask3.git
cd HHgoaTask3

# Create a virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and configure your API keys and RPC endpoints:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
# SerpAPI Key for Live Google Lens Reverse Search (Optional: Fallback activated if empty)
SERPAPI_KEY=your_serpapi_api_key_here

# EVM RPC Endpoint (e.g. Ethereum Sepolia, Polygon Amoy, or local node)
RPC_URL=https://ethereum-sepolia-rpc.publicnode.com

# Private Key for signing testnet transactions (0x...)
PRIVATE_KEY=0x_your_testnet_private_key_here

# Web Server Port
PORT=5000
```

---

## 🚀 Usage Guide

### Option A: Launch Interactive Web UI

Launch the Flask dashboard to perform face scans in your web browser:

```bash
python run.py --ui
```
Open **`http://127.0.0.1:5000`** in your browser to:
- Drag-and-drop facial scan images.
- Inspect crop bounding boxes and biometric embeddings.
- View live social media profile matches and similarity confidence scores.
- Trigger real-time blockchain settlement and inspect transaction calldata proofs.

### Option B: Run CLI Pipeline

Process an image directly via the command line:

```bash
# Run pipeline on test image
python run.py --image assets/test.jpg

# Save output to a custom directory
python run.py --image assets/sample_face.jpg --output output_results
```

### Option C: Independent On-Chain Verification

Independently audit any previously registered blockchain transaction hash:

```bash
python run.py --verify-tx 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
```

---

## 🧪 Testing & Verification

Run the full automated PyTest suite to verify system integrity:

```bash
python -m pytest tests/ -v
```

### Test Coverage Highlights:
- **`test_face_detector.py`**: Validates face detection, crop output, embedding dimensions (128-d), and SHA-256 biometric hashing.
- **`test_web_searcher.py`**: Tests social platform filtering, cosine similarity calculation accuracy, and mock search fallback mode.
- **`test_blockchain_verifier.py`**: Tests canonical proof state hashing, payload calldata formatting (`0x66616365...`), and verification logic.
- **`test_pipeline.py`**: End-to-end integration tests verifying sequential execution from image file to blockchain proof.

---

## 🔐 Cryptographic Security & Privacy

1. **Zero Raw PII On-Chain**: Raw facial images are never stored on public blockchains. Only L2-normalized biometric embedding hashes and canonical verification state proofs are anchored.
2. **Canonical State Fingerprint**:
   $$\text{StateHash} = \text{SHA256}(\text{BiometricHash} \mathbin{\Vert} \text{PostURL} \mathbin{\Vert} \text{Platform} \mathbin{\Vert} \text{Author} \mathbin{\Vert} \text{Timestamp})$$
3. **Calldata Anchoring**: Hex-encoded verification metadata is stored directly in transaction `calldata`, creating an unalterable, timestamped audit trail on EVM blockchains.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.
