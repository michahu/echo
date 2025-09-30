#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of scripts)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to run evaluation on a specific directory
run_eval_on_dir() {
  local dir_path="$1"
  if [ -d "$dir_path" ]; then
    echo "Running evals for $dir_path"
    (cd "$PROJECT_ROOT" && uv run python -m src.experimentation.run_evals \
      --saved_outputs_folder "$dir_path" \
      --metrics_file "$dir_path/metrics.json" \
      --metric_names task_efficiency checklist reference_overlap_using_llm)
  fi
}

# Function to recursively find and evaluate directories
evaluate_recursively() {
  local base_dir="$1"
  local depth="$2"
  
  # Check if this directory has subdirectories
  local has_subdirs=false
  for item in "$base_dir"/*; do
    if [ -d "$item" ]; then
      has_subdirs=true
      break
    fi
  done
  
  # If no subdirectories, this might be a leaf directory to evaluate
  if [ "$has_subdirs" = false ]; then
    # Check if this looks like an evaluation target (contains expected files/structure)
    if [ -d "$base_dir" ]; then
      echo "Found evaluation target: $base_dir"
      run_eval_on_dir "$base_dir"
    fi
    return
  fi
  
  # Recurse into subdirectories
  for dir in "$base_dir"/*/; do
    if [ -d "$dir" ]; then
      dir_name=${dir%/}
      evaluate_recursively "$dir_name" $((depth + 1))
    fi
  done
}

BASE_DIR=$1

# Start recursive evaluation
echo "Starting recursive evaluation from: $BASE_DIR"
evaluate_recursively "$BASE_DIR" 0