#!/usr/bin/env python3
"""
Pre-commit checks script for aind-hcr-data-loader
Run this before submitting a pull request
"""

import os
import subprocess
import sys


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✅ {description} complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Main function to run all pre-commit checks."""
    print("🔍 Running pre-commit checks...")
    print("=" * 50)

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {os.getcwd()}")
    print()

    checks = [
        ("black .", "Running black to format code"),
        ("isort .", "Running isort to sort imports"),
        ("flake8 .", "Running flake8 for linting"),
        ("interrogate .", "Running interrogate for documentation coverage"),
        ("coverage run -m unittest discover && coverage report", "Running tests with coverage"),
    ]

    all_passed = True

    for command, description in checks:
        if not run_command(command, description):
            all_passed = False
            break
        print()

    if all_passed:
        print("🎉 All pre-commit checks passed!")
        print("Your code is ready for pull request submission.")
        sys.exit(0)
    else:
        print("💥 Some checks failed. Please fix the issues and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
