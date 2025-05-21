#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 job_id"
    exit 1
fi

# Assign the job ID from the command-line arguments
JOB_ID=$1

# Define the top-level directory
TOP_DIR="${JOB_ID}"

# Check if the top-level directory exists
if [ ! -d "$TOP_DIR" ]; then
    echo "Directory $TOP_DIR does not exist."
    exit 1
fi

BEST_CKPT=""
BEST_SUCCESS_RATE=0

# Iterate over each subdirectory in TOP_DIR
for CKPT_DIR in "$TOP_DIR"/*; do
    CKPT_ID=$(basename "$CKPT_DIR")
    # Find the .txt file in CKPT_DIR with the name pattern "{success_rate}%.txt"
    SUCCESS_FILE=$(find "$CKPT_DIR" -maxdepth 1 -type f -name "*.txt" | grep -E "[0-9]+\.[0-9]+%\.txt")
    
    if [ -z "$SUCCESS_FILE" ]; then
        continue
    fi
    
    # Extract the success rate from the filename
    SUCCESS_RATE=$(basename "$SUCCESS_FILE" .txt | sed 's/%//')
    
    # Compare and update best checkpoint
    if (( $(echo "$SUCCESS_RATE > $BEST_SUCCESS_RATE" | bc -l) )); then
        BEST_SUCCESS_RATE=$SUCCESS_RATE
        BEST_CKPT=$CKPT_ID
    fi

done

if [ -z "$BEST_CKPT" ]; then
    echo "No valid success rate files found in any checkpoint directories."
    exit 1
fi

echo "$BEST_SUCCESS_RATE $BEST_CKPT"
