#!/bin/bash
echo "Running run_model.sh"

if [ $1 = "train" ]; then
    TRAIN_DIR=$2
    echo "Training"

    python -m src.train "$TRAIN_DIR"

elif [ $1 = "eval" ]; then 
    TEST_DIR=$2

    python -m src.eval "$TEST_DIR"

fi