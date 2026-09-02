# HH Goa 2026 Task 3: Face Identification and Blockchain Verification Pipeline

> An end-to-end, tamper-evident autonomous pipeline that takes a face scan image, discovers matching social media posts, and uploads a cryptographic proof onto an EVM blockchain for independent re-verification.

## 1. Overview and Architecture

`
+----------------------------+     +----------------------------+     +----------------------------+
| STAGE 1: Face Detection    |---->| STAGE 2: Web/Social Search |---->| STAGE 3: Blockchain Verify  |
+----------------------------+     +----------------------------+     +----------------------------+
`

### Stage 1: Face Detection and Feature Encoding
- Deep-learning YuNet detector for face localization and bounding box extraction.
- SFace 128-d normalized facial feature vector embeddings.
- Deterministic SHA-256 biometric face hash.

### Stage 2: Web and Social Media Search
- SerpAPI (Google Lens) reverse image discovery for live social media matching (Twitter/X, LinkedIn, GitHub, Instagram, Reddit).
- Biometric facial cosine similarity verification with match confidence percentage.
- Built-in zero-dependency search fallback for offline testing.

### Stage 3: Blockchain Upload and Re-Verification
- Canonical SHA-256 fingerprinting binding face scan, post URL, author, and timestamp.
- Broadcasts on-chain transactions to Ethereum Sepolia / Polygon Amoy testnet via web3.py with calldata embedding.
- Independent tamper-evidence verification proves authenticity and rejects modified records.

## 2. Quick Start

```powershell
pip install -r requirements.txt

# Option A: Run Interactive HH Goa Themed Web UI
python run.py --ui
# (Open http://127.0.0.1:5000 in your browser)

# Option B: Run CLI Pipeline Directly
python run.py --image assets/test.jpg

# Run Automated Test Suite (17 Tests)
python -m pytest tests/ -v
```

## 3. Configuration (.env)

`env
SERPAPI_KEY=your_serpapi_key
RPC_URL=https://ethereum-sepolia-rpc.publicnode.com
PRIVATE_KEY=0x_your_testnet_private_key
`

## 4. Known Limitations
- SerpAPI free tier quota (250 searches/month).
- Testnet gas fees require free faucet ETH.
