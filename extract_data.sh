#!/bin/bash

# Check for the required number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: ./convert_to_wav.sh <source_directory> <destination_directory>"
    exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg could not be found. Please install it first."
    exit 1
fi

SRC_DIR="$1"
DEST_DIR="$2"

# Create the destination directories if they don't exist
mkdir -p "$DEST_DIR/audio"
mkdir -p "$DEST_DIR/video"

# Convert MP4 to WAV and copy MP4 files
for file in "$SRC_DIR"/*.mp4; do
    # Extract the filename without extension
    filename=$(basename -- "$file")
    base="${filename%.*}"

    # Convert the audio to WAV format
    ffmpeg -i "$file" -vn -acodec pcm_s16le -ac 2 -ar 44100 "$DEST_DIR/audio/$base.wav"
   
    # Copy the MP4 file to the video directory
    mv "$file" "$DEST_DIR/video/$filename"
done

echo "Conversion completed."

