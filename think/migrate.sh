#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 DIRECTORY" >&2
  exit 1
fi

base_dir="$1"

if [ ! -d "$base_dir" ]; then
  echo "Directory not found: $base_dir" >&2
  exit 1
fi

# Move audio files: audio_YYYYMMDD_HHMMSS.{ogg,json}
for path in "$base_dir"/audio_*_*.*; do
  [ -e "$path" ] || continue
  name="$(basename "$path")"
  if [[ $name =~ ^audio_([0-9]{8})_([0-9]{6})\.(ogg|json)$ ]]; then
    day="${BASH_REMATCH[1]}"
    time="${BASH_REMATCH[2]}"
    ext="${BASH_REMATCH[3]}"
    dest_dir="$base_dir/$day"
    mkdir -p "$dest_dir"
    new_path="$dest_dir/${time}_audio.$ext"
    if [ -e "$new_path" ]; then
      echo "Skipping $path, target exists" >&2
    else
      echo "Moving $path -> $new_path"
      mv "$path" "$new_path"
    fi
  fi
done

# Move monitor diff files: monitor_INDEX_YYYYMMDD_HHMMSS_diff.{png,json}
for path in "$base_dir"/monitor_*_*_diff.*; do
  [ -e "$path" ] || continue
  name="$(basename "$path")"
  if [[ $name =~ ^monitor_([0-9]+)_([0-9]{8})_([0-9]{6})_diff\.(png|json)$ ]]; then
    idx="${BASH_REMATCH[1]}"
    day="${BASH_REMATCH[2]}"
    time="${BASH_REMATCH[3]}"
    ext="${BASH_REMATCH[4]}"
    dest_dir="$base_dir/$day"
    mkdir -p "$dest_dir"
    new_path="$dest_dir/${time}_monitor_${idx}_diff.$ext"
    if [ -e "$new_path" ]; then
      echo "Skipping $path, target exists" >&2
    else
      echo "Moving $path -> $new_path"
      mv "$path" "$new_path"
    fi
  fi
done

