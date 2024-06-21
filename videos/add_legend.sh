#!/bin/bash
# Script to add multiple text overlays to a video using FFmpeg

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <input_video_file>"
  exit 1
fi

# Assign the first argument to a variable
INPUT_VIDEO=$1
OUTPUT_VIDEO="${INPUT_VIDEO%.*}-legended.mkv"

# Define the common parameters
FONTFILE="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONTCOLOR="white"
FONTSIZE=20
BOXCOLOR="green@1"
BOXBORDER=3

# Define the texts and their y positions
TEXTS=("Model 4 (Big Batch, 68M)" "Model 5 (Big Batch, Dynamic LR, 68M)" "Model 6 (Big Batch, Entropy, 68M)" "Model 7 (Dynamic Batch and LR, 68M)" "Baseline (68M)" "Baseline pretrained (439M)")
XPOS=(5 490 975 5 490 975)
YPOS=(5 5 5 440 440 440)

# Start building the drawtext filter
FILTER=""

for i in ${!TEXTS[@]}; do
  FILTER+="drawtext=fontfile=$FONTFILE:text='${TEXTS[$i]}':fontcolor=$FONTCOLOR:fontsize=$FONTSIZE:x=${XPOS[$i]}:y=${YPOS[$i]}:box=1:boxcolor=$BOXCOLOR:boxborderw=$BOXBORDER,"
done

# Remove the trailing comma
FILTER=${FILTER%,}

# Run FFmpeg with the generated filter
ffmpeg -i "$INPUT_VIDEO" -vf "$FILTER" -codec:a copy "$OUTPUT_VIDEO"


