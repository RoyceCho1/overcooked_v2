#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <run_directory> [destination_directory]"
  exit 1
fi

# Set the source directory
# Set the source directory
if [ -d "$1" ]; then
  SRC_DIR="$1"
  RUN_NAME=$(basename "$1")
else
  SRC_DIR="runs/$1"
  RUN_NAME="$1"
fi

# Set the destination directory
if [ -z "$2" ]; then
  DEST_DIR="fcp_populations/$RUN_NAME"
else
  DEST_DIR="$2/$RUN_NAME"
fi

# Create the destination directory
mkdir -p $DEST_DIR

# Initialize variables
counter=0
subdir_counter=0

# Loop over the runs
for run in "$SRC_DIR"/run_*; do
  # Check if we need to create a new subdirectory
  if (( counter % 8 == 0 )); then
    subdir_counter=$((subdir_counter + 1))
    mkdir -p "$DEST_DIR/fcp_$subdir_counter"
  fi
  
  # Copy the run to the appropriate subdirectory
  cp -r "$run" "$DEST_DIR/fcp_$subdir_counter/"
  
  # Increment the counter
  counter=$((counter + 1))
done
