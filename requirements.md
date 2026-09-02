# Requirements — HH Goa 2026 Task 3: Face Identification & Blockchain Verification

## 1. Overview

Build an **end-to-end pipeline** that:

1. Takes a **face scan** (image) as input
2. **Searches the web / social media** to find matching posts
3. **Uploads & verifies** the discovered data on a **blockchain**

```
Face scan input → Web/social media search → Blockchain upload/verification
```

> [!IMPORTANT]
> No website/UI is required. Focus entirely on the **pipeline** itself.

---

## 2. Functional Requirements

### FR-1 · Face Detection & Encoding
| Aspect | Detail |
|--------|--------|
| **Input** | A single image containing a human face |
| **Output** | A face encoding / feature vector suitable for comparison |
| **Constraint** | Any face detection/recognition library or API is acceptable (e.g. `face_recognition`, `dlib`, `DeepFace`, `InsightFace`, cloud APIs) |

### FR-2 · Social Media / Web Search
| Aspect | Detail |
|--------|--------|
| **Input** | The detected face (or source image) |
| **Output** | At least **one real, matching social-media post** (image, text, URL, metadata) |
| **Constraint** | Must be a **genuine search step** — no hardcoded or pre-picked results |
| **Acceptable approaches** | Reverse image search (Google, Yandex, SerpAPI, etc.), social-media API scraping, or scripted search |

### FR-3 · Blockchain Verification
| Aspect | Detail |
|--------|--------|
| **Input** | The matched post data (image, text, or metadata) |
| **Process** | Hash / fingerprint the data and **upload to a blockchain** |
| **Output** | An on-chain record (transaction hash / smart-contract entry) that can be independently **re-verified** |
| **Constraint** | Any blockchain is acceptable — public testnet, mainnet, or local/simulated chain |
| **Key criterion** | Must demonstrate **tamper-evidence**: re-verifying the data against the on-chain record |

### FR-4 · End-to-End Pipeline Execution
- The three stages above must connect into a **single runnable pipeline**
- Should be demonstrable in one continuous screen recording

---

## 3. Non-Functional Requirements

| ID | Requirement | Detail |
|----|-------------|--------|
| **NFR-1** | **GitHub Repo** | Full source code in a public GitHub repository |
| **NFR-2** | **README** | Must cover: what the project does, how to run it, which blockchain was used, known limitations |
| **NFR-3** | **Screen Recording** | End-to-end demo: face scan → social post found → blockchain upload/verification. Plain recording, no editing needed. Hosted on YouTube (unlisted), Google Drive, Loom, etc. |
| **NFR-4** | **No Website** | No hosted web app required |
| **NFR-5** | **Single Submission** | No resubmissions allowed — submit only when final |

---

## 4. Submission Deliverables

| # | Deliverable | Notes |
|---|-------------|-------|
| 1 | **GitHub repo link** | Public, with README |
| 2 | **Screen recording link** | Publicly accessible |
| 3 | **Submission form** | [Google Form](https://forms.gle/oZbQGuwiNeHVcHWo8) |

---

## 5. Timeline

| Milestone | Date |
|-----------|------|
| Task launch | **August 31, 2026** |
| Submission deadline | **September 7, 2026 — 11:59 PM** |

> [!CAUTION]
> **7-day window. No resubmissions.** The build must be final before submitting.

---

## 6. Acceptance Criteria

- [ ] Face is correctly detected and encoded from an input image
- [ ] At least one **real** matching social-media post is found via genuine search
- [ ] Post data (or its hash) is uploaded to a blockchain
- [ ] The on-chain record can be **re-verified** to prove tamper-evidence
- [ ] GitHub repo is public with a complete README
- [ ] Screen recording shows the full pipeline working end-to-end
- [ ] Submission form is filled and submitted before the deadline
