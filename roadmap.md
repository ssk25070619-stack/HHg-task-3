# Roadmap — HH Goa 2026 Task 3: Face Identification & Blockchain Verification

> **Timeline:** Aug 31 – Sep 7, 2026 (7 days)  
> **Goal:** End-to-end pipeline → Face scan → Social media match → Blockchain verification

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  STAGE 1        │     │  STAGE 2             │     │  STAGE 3            │
│  Face Detection │────▶│  Web/Social Search   │────▶│  Blockchain Upload  │
│  & Encoding     │     │  & Match             │     │  & Verification     │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘
   Input: Image            Input: Face/Image           Input: Post data
   Output: Encoding        Output: Matched post        Output: TX hash + proof
```

---

## Tech Stack (Proposed)

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Language** | Python 3.11+ | Rich ML/crypto ecosystem |
| **Face Detection** | `face_recognition` / `DeepFace` / `InsightFace` | Battle-tested, easy to use |
| **Web Search** | SerpAPI (Google reverse image) / Yandex reverse image / `icrawler` | Genuine search, not hardcoded |
| **Blockchain** | Ethereum Sepolia testnet via `web3.py` **OR** Polygon Amoy testnet **OR** Solana devnet | Free testnet, easy to demo re-verification |
| **Hashing** | SHA-256 of post content/metadata | Standard, tamper-evident |
| **CLI/Runner** | Python CLI (`argparse` / `click`) or simple script | No website needed |

> [!NOTE]
> Tech choices are flexible — finalize during Day 1 research. The above is a starting recommendation.

---

## Phase Breakdown

### Phase 0 — Setup & Research *(Day 1: Aug 31)*

| Task | Details | Output |
|------|---------|--------|
| Initialize GitHub repo | Create repo, `.gitignore`, virtual env, `requirements.txt` | Clean repo skeleton |
| Research face-recognition libs | Compare `face_recognition`, `DeepFace`, `InsightFace` — pick one | Decision documented |
| Research reverse-image-search APIs | Evaluate SerpAPI, Yandex, Google Lens API, social scraping tools | API key obtained, approach chosen |
| Research blockchain options | Compare Sepolia, Polygon Amoy, Solana devnet, Hyperledger | Testnet chosen, wallet funded |
| Define project structure | Modular layout (see below) | Folder structure created |

**Proposed project structure:**
```
HHGoa-task-3/
├── src/
│   ├── face_detector.py       # Stage 1: Face detection & encoding
│   ├── web_searcher.py        # Stage 2: Reverse image / social search
│   ├── blockchain_verifier.py # Stage 3: Blockchain upload & verification
│   └── pipeline.py            # Orchestrator — ties all stages together
├── contracts/                 # Smart contract (if using Ethereum/Solidity)
│   └── DataVerifier.sol
├── tests/
│   ├── test_face.py
│   ├── test_search.py
│   └── test_blockchain.py
├── assets/
│   └── sample_face.jpg        # Test input image
├── .env.example               # Template for API keys
├── requirements.txt
├── README.md
└── run.py                     # Entry point
```

---

### Phase 1 — Face Detection & Encoding *(Day 2: Sep 1)*

| Task | Details | Done Criteria |
|------|---------|---------------|
| Implement face detection | Load image → detect face region → crop | Face bounding box drawn correctly |
| Implement face encoding | Generate 128-d (or similar) face embedding | Encoding vector produced for any input face |
| Handle edge cases | No face found, multiple faces, low quality | Graceful error messages |
| Write unit tests | Test with known images | Tests pass |

**Key decisions:**
- Single face vs. multi-face: Pick the **largest/most prominent** face if multiple detected
- Output format: Save encoding as `.npy` or keep in-memory for pipeline

---

### Phase 2 — Web / Social Media Search *(Days 3–4: Sep 2–3)*

> [!IMPORTANT]
> This is the **hardest and most unpredictable** stage. Allocate extra time.

| Task | Details | Done Criteria |
|------|---------|---------------|
| Implement reverse image search | Upload face image → get search results via API | At least 1 result returned |
| Parse & filter results | Extract post URL, image, text, metadata from results | Structured data object |
| Match verification | Compare found face with input face encoding (similarity threshold) | Confidence score logged |
| Handle search failures | Rate limits, no results, API errors | Fallback logic + clear error reporting |
| Test with real faces | Use public figures / your own social media | End-to-end search works |

**Approach options (pick one or combine):**

| Approach | Pros | Cons |
|----------|------|------|
| **SerpAPI** (Google reverse image) | Reliable, structured JSON | Paid (free tier: 100 searches/month) |
| **Yandex reverse image** | Free, good for faces | Less structured, may need scraping |
| **Social media APIs** (Twitter/X, Instagram) | Direct platform data | Rate limits, auth complexity |
| **Custom scraping** | Free, flexible | Fragile, ToS concerns |

---

### Phase 3 — Blockchain Upload & Verification *(Days 4–5: Sep 3–4)*

| Task | Details | Done Criteria |
|------|---------|---------------|
| Set up blockchain connection | Connect to testnet via `web3.py` or equivalent SDK | Successful connection |
| Create wallet | Generate or import wallet, fund with testnet ETH/tokens | Wallet funded |
| Write smart contract (if Ethereum) | Simple `store(bytes32 hash)` + `verify(bytes32 hash)` | Contract deployed to testnet |
| Implement upload function | Hash post data → call contract / send TX | TX hash returned |
| Implement verify function | Given original data → recompute hash → check against on-chain record | Returns `True/False` match |
| Test tamper detection | Modify data slightly → re-verify → should fail | Tamper detected correctly |

**Blockchain options comparison:**

| Chain | SDK | Cost | Speed | Difficulty |
|-------|-----|------|-------|------------|
| **Ethereum Sepolia** | `web3.py` | Free (faucet) | ~15s blocks | Medium |
| **Polygon Amoy** | `web3.py` | Free (faucet) | ~2s blocks | Medium |
| **Solana Devnet** | `solana-py` | Free | ~400ms | Higher |
| **Local Ganache** | `web3.py` | Free | Instant | Easy (but less impressive) |

---

### Phase 4 — Pipeline Integration *(Day 5: Sep 4)*

| Task | Details | Done Criteria |
|------|---------|---------------|
| Wire all 3 stages | `pipeline.py` orchestrates: detect → search → upload | Single command runs everything |
| Add CLI interface | `python run.py --image face.jpg` | Clean CLI with args |
| Add logging | Each stage logs progress, results, timing | Clear console output |
| Error handling | Graceful failures at each stage with informative messages | No uncaught exceptions |
| End-to-end test | Run full pipeline with a real face image | TX hash produced, verification passes |

---

### Phase 5 — Documentation & Recording *(Day 6: Sep 5)*

| Task | Details | Done Criteria |
|------|---------|---------------|
| Write README.md | What it does, setup instructions, blockchain used, limitations | Complete, clear README |
| Add `.env.example` | Template for all required API keys/secrets | All vars documented |
| Record screen demo | Full pipeline: face input → search → blockchain → verify | Unedited screen recording |
| Upload recording | YouTube (unlisted) / Google Drive / Loom | Shareable link obtained |
| Clean up repo | Remove debug code, organize files, final commit | Repo is presentable |

---

### Phase 6 — Buffer & Submission *(Day 7: Sep 6–7)*

| Task | Details | Done Criteria |
|------|---------|---------------|
| Final testing | Run pipeline 2-3 times with different faces | All runs succeed |
| Fix any remaining issues | Bugs, edge cases, documentation gaps | Issues resolved |
| **Submit** | Fill Google Form with repo + recording links | ✅ Submitted |

> [!CAUTION]
> **No resubmissions allowed.** Do NOT submit until everything is verified and final.

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Reverse image search API returns no matches | Pipeline breaks at Stage 2 | **High** | Test with well-known public figures first; have backup search API |
| API rate limits exceeded | Search stage fails | Medium | Cache results; use multiple API keys; test sparingly |
| Blockchain testnet is slow/down | Can't demo Stage 3 | Low | Have local Ganache as fallback |
| Face detection fails on certain images | Stage 1 fails | Low | Support multiple input formats; pre-validate image quality |
| Running out of time | Incomplete submission | Medium | Prioritize MVP (all 3 stages working) over polish |

---

## MVP vs. Stretch Goals

### MVP (Must Have)
- [ ] Face detected from image
- [ ] At least 1 real social-media match found
- [ ] Match data hashed and uploaded to blockchain
- [ ] On-chain verification succeeds
- [ ] GitHub repo with README
- [ ] Screen recording of full pipeline

### Stretch Goals (Nice to Have)
- [ ] Web UI (Streamlit/Gradio) for demo — *not required but impressive*
- [ ] Multi-face support — detect and search multiple faces
- [ ] Confidence scoring — show face-match similarity percentage
- [ ] IPFS storage — store full post data on IPFS, hash on blockchain
- [ ] Batch processing — run pipeline on multiple images
- [ ] Verification dashboard — query on-chain records by hash

---

## Daily Schedule Summary

| Day | Date | Focus | Milestone |
|-----|------|-------|-----------|
| 1 | Aug 31 (Sun) | Setup, research, decisions | Repo created, tech stack locked |
| 2 | Sep 1 (Mon) | Face detection & encoding | Stage 1 complete |
| 3 | Sep 2 (Tue) | Web/social search | Stage 2 prototype |
| 4 | Sep 3 (Wed) | Search polish + blockchain start | Stage 2 done, Stage 3 started |
| 5 | Sep 4 (Thu) | Blockchain + pipeline integration | End-to-end pipeline works |
| 6 | Sep 5 (Fri) | Documentation + recording | README + video done |
| 7 | Sep 6–7 (Sat–Sun) | Buffer + final submit | ✅ **SUBMITTED** |
