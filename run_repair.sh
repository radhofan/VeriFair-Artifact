#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: bash run_repair.sh [AC|BM|GC|DF|UCI]"
  exit 1
fi

DATASET=$1

echo "====================================="
echo "Running VeriFair pipeline for: $DATASET"
echo "====================================="

# Base directory
BASE_DIR="VeriFair/src"

# Define commands per dataset
case "$DATASET" in
  "AC")
    echo "Running Adult (AC) Fairness Evaluation..."
    python $BASE_DIR/AC/train_fair_model.py
    python $BASE_DIR/AC/metric_aif360.py
    python $BASE_DIR/AC/metric_themis_causality.py
    python $BASE_DIR/AC/metric_random_unfairness.py
    ;;
  "BM")
    echo "Running Bank Marketing (BM) Fairness Evaluation..."
    python $BASE_DIR/BM/train_fair_model.py
    python $BASE_DIR/BM/metric_aif360.py
    python $BASE_DIR/BM/metric_themis_causality.py
    python $BASE_DIR/BM/metric_themis_causality.py
    python $BASE_DIR/BM/metric_random_unfairness.py
    ;;
  "GC")
    echo "Running German Credit (GC) Fairness Evaluation..."
    python $BASE_DIR/GC/train_fair_model.py
    python $BASE_DIR/GC/metric_aif360.py
    python $BASE_DIR/GC/metric_themis_causality.py
    python $BASE_DIR/GC/metric_random_unfairness.py
    ;;
  "DF")
    echo "Running Default (DF) Fairness Evaluation..."
    python $BASE_DIR/DF/train_fair_model.py
    python $BASE_DIR/DF/metric_aif360.py
    python $BASE_DIR/DF/metric_themis_causality.py
    python $BASE_DIR/DF/metric_random_unfairness.py
    ;;
  "UCI")
    echo "Running UCI Fairness Evaluation..."
    python $BASE_DIR/UCI/train_fair_model.py
    python $BASE_DIR/UCI/metric_aif360.py
    python $BASE_DIR/UCI/metric_themis_causality.py
    python $BASE_DIR/UCI/metric_random_unfairness.py
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    echo "Available options: AC, BM, GC, DF, UCI"
    exit 1
    ;;
esac

echo "====================================="
echo "✅ All tasks completed for $DATASET!"
echo "====================================="
