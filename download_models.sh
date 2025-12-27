#!/usr/bin/env bash

set -e  # exit on error

# -----------------------------
# Config
# -----------------------------
MODEL_URL="https://vandanresearch.sgp1.digitaloceanspaces.com/shallow-parsers/models.zip"
BASE_DIR="$(pwd)"
TARGET_DIR="$BASE_DIR"
ZIP_NAME="models.zip"

# -----------------------------
# Create target directory
# -----------------------------
echo "Creating model directory..."
mkdir -p "$TARGET_DIR"

# -----------------------------
# Download
# -----------------------------
if [ ! -f "$ZIP_NAME" ]; then
    echo "Downloading models..."
    curl -L "$MODEL_URL" -o "$ZIP_NAME"
else
    echo "models.zip already exists, skipping download"
fi

# -----------------------------
# Extract
# -----------------------------
echo "Extracting models..."
unzip -o "$ZIP_NAME" -d "$TARGET_DIR"

echo "All models installed correctly!"

# -----------------------------
# Cleanup (optional)
# -----------------------------
echo "leaning up zip file..."
rm -f "$ZIP_NAME"

echo "Done!"

