#!/bin/bash
# Script to convert video to gif using ffmpeg

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <input_video_file>"
  exit 1
fi

INPUT=$1
OUTPUT="${INPUT_VIDEO%.*}.gif"

ffmpeg -ss 00:00:00 -t 11 -i "$INPUT" -vf "fps=30,scale=800:-1:flags=lanczos" -gifflags +transdiff -y "$OUTPUT"
