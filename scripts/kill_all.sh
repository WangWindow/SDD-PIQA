#!/bin/bash

echo "Stopping SDD-PIQA related processes..."

# Kill Python processes
# Matches: python -u utils/train_recognition/train_recognition.py
pkill -f "utils/train_recognition/train_recognition.py"
# Matches: python -u utils/gen_pseudo_labels/*.py
pkill -f "utils/gen_pseudo_labels/"
# Matches: python -u train.py
pkill -f "python -u train.py"

# Kill Shell scripts
pkill -f "scripts/run_all.sh"
pkill -f "scripts/train_rec.sh"
pkill -f "scripts/gen_pseudo_labels.sh"
pkill -f "scripts/train.sh"

echo "All matching processes have been signaled to stop."
