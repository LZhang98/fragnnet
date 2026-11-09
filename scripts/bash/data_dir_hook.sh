#!/bin/bash
set -euo pipefail

# local data dir = ~/fragnnet/data/
LOCAL_DIR="$HOME/fragnnet/data_v2"

# symlink target = /mnt/data/masster_data/
TARGET_DIR="/mnt/data/masster_data"

# subdirs to link: df, frag, misc, proc, raw, split
SUBDIRS=(df frag misc proc raw split)

mkdir -p "$LOCAL_DIR"

for sub in "${SUBDIRS[@]}"; do
    link="$LOCAL_DIR/$sub"
    target="$TARGET_DIR/$sub"

    if [ -L "$link" ]; then
        # existing symlink: repoint it
        ln -sfn "$target" "$link"
    elif [ -e "$link" ]; then
        echo "WARNING: $link exists and is not a symlink; skipping" >&2
        continue
    else
        ln -s "$target" "$link"
    fi
    echo "$link -> $target"
done
