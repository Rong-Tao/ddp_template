#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Find the most recently created TensorBoard directory
latest_dir=$(find . -name "tensorboard" -type d -printf '%T+ %p\n' | sort -r | head -n 1 | cut -d' ' -f2-)

if [ -z "$latest_dir" ]; then
    echo "No TensorBoard directory found."
    exit 1
fi

echo "Latest TensorBoard directory: $latest_dir"

# Run TensorBoard
tensorboard --logdir="$latest_dir" --port=0
