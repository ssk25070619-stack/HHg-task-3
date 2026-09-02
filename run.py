#!/usr/bin/env python3
"""
HH Goa Task 3 - CLI Entry Point
Run the face detection and verification pipeline from CLI.

Usage:
    python run.py --image assets/sample_face.jpg
"""

import sys
import os

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
from src.pipeline import VerificationPipeline

def main():
    parser = argparse.ArgumentParser(description="HH Goa 2026 Task 3: Face Identification & Blockchain Verification Pipeline")
    parser.add_argument("--image", "-i", type=str, help="Path to input image containing a face scan")
    parser.add_argument("--output", "-o", type=str, default="output", help="Directory to save cropped face and pipeline outputs")
    parser.add_argument("--verify-tx", type=str, help="Transaction hash to independently re-verify on blockchain")
    parser.add_argument("--ui", action="store_true", help="Launch interactive HH Goa themed Web UI")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Web UI server (default: 5000)")

    args = parser.parse_args()

    if args.ui:
        from app import app
        print("\n==================================================")
        print(f" [HH GOA 2026] Launching Web UI at http://127.0.0.1:{args.port}")
        print("==================================================\n")
        app.run(host="0.0.0.0", port=args.port, debug=False)
        sys.exit(0)

    if args.verify_tx:
        from src.blockchain_verifier import BlockchainVerifier
        verifier = BlockchainVerifier()
        print("==================================================")
        print(f" Querying Blockchain for TX: {args.verify_tx}")
        print("==================================================")
        res = verifier.verify_record({}, args.verify_tx)
        print(f"Status: {res.get('status')}")
        print(f"Verified: {res.get('is_verified')}")
        print(f"Details: {res}")
        sys.exit(0)

    if not args.image:
        print("Error: Please provide an input image path using --image or -i, or launch UI with --ui.")
        print("Examples:")
        print("    python run.py --image assets/sample_face.jpg")
        print("    python run.py --ui")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"Error: Specified image path does not exist: {args.image}")
        sys.exit(1)

    pipeline = VerificationPipeline()
    result = pipeline.run(image_path=args.image, output_dir=args.output)
    
    if result.get("success"):
        print("\n[OK] Pipeline execution completed successfully.")
    else:
        print("\n[ERROR] Pipeline execution encountered an error.")
        sys.exit(1)

if __name__ == "__main__":
    main()
