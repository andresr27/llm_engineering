#!/usr/bin/env python3
"""Run TTS tests for the AI Tutor Chatbot."""

import subprocess
import sys

def run_pytest():
    """Run pytest with the appropriate options."""
    cmd = [
        sys.executable, "-m", "pytest",
        "test_tts.py",
        "-v",  # Verbose output
        "-p", "no:hydra",  # Disable hydra plugin to avoid Python 3.12 compatibility issues
        "--tb=short",  # Short tracebacks
        "--disable-warnings",  # Disable warnings for cleaner output
        # Remove the -k filter to run all tests including RAG
    ]
    
    print("Running TTS tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code: {e.returncode}")
        return False

if __name__ == "__main__":
    success = run_pytest()
    sys.exit(0 if success else 1)
