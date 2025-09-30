#!/bin/bash

# Script to run experiments on all JSON files
# Usage: ./run_stateful_experiments.sh [num_files_per_line] [port] [alg_name] [task_name] [history]
# Runs experiments for all JSON files in the directory
# If num_files_per_line is provided, only that many files per line will be run.
# If lines is provided, only those specific line numbers will be run (comma-separated list or range like 1-3)

NUM_FILES_TO_RUN=$1
PORT=$2
ALG_NAME=$3
TASK_NAME=$4
HISTORY=$5
LINE_NUMS=$6


EXPERIMENT_ORDER_FILE=workspace/peoplejoin-qa/experiments/exp_configs_test/$TASK_NAME/experiment_order.txt
OUTPUT_DIR=workspace/stateful-no-history/$ALG_NAME/$HISTORY/$TASK_NAME/
AGENT_CONFIG_PATH=workspace/peoplejoin-qa/experiments/agent_configs/agentconf_${TASK_NAME}_gpt4o_oneexample.json

# Function to check if a line number should be processed
should_process_line() {
    local line_num=$1
    
    # If LINE_NUMS is not set, process all LINE_NUMS
    if [ -z "$LINE_NUMS" ]; then
        return 0
    fi
    
    # Handle comma-separated list and ranges
    IFS=',' read -ra LINE_SPECS <<< "$LINE_NUMS"
    for spec in "${LINE_SPECS[@]}"; do
        if [[ "$spec" == *"-"* ]]; then
            # Handle range (e.g., 1-3)
            IFS='-' read -ra RANGE <<< "$spec"
            start=${RANGE[0]}
            end=${RANGE[1]}
            if [ "$line_num" -ge "$start" ] && [ "$line_num" -le "$end" ]; then
                return 0
            fi
        else
            # Handle single number
            if [ "$line_num" -eq "$spec" ]; then
                return 0
            fi
        fi
    done
    
    return 1
}

# uv run python -m src.experimentation.experiment_with_hitl_or_simulation workspace/peoplejoin-qa/experiments/exp_configs_test/driving_school/0.json . --agent_config_path workspace/peoplejoin-qa/experiments/agent_configs/agentconf_driving_school_gpt4o_oneexample.json  --load_pth . --port 54077


# Check if experiment order file exists
if [ ! -f "$EXPERIMENT_ORDER_FILE" ]; then
    echo "Error: File $EXPERIMENT_ORDER_FILE does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Read experiment order file into an array of lines
mapfile -t lines < "$EXPERIMENT_ORDER_FILE"


# Show which lines will be processed
if [ -n "$LINE_NUMS" ]; then
    echo "Will process only lines: $LINE_NUMS"
else
    echo "Will process all ${#lines[@]} lines"
fi

# Process each line
for i in "${!lines[@]}"; do
    line_num=$((i + 1))
    line="${lines[$i]}"
    
    # Check if this line should be processed based on LINES parameter
    if ! should_process_line "$line_num"; then
        echo "Skipping line $line_num (not in specified LINES: $LINES)"
        continue
    fi
    
    LINE_OUTPUT_DIR="$OUTPUT_DIR/$line_num"
    echo "Processing line $line_num: $line"
    
    # Each line contains a space-separated list of JSON files
    JSON_FILES=($line)
    
    # If NUM_FILES_TO_RUN is set and not 'max', slice the array
    if [ -n "$NUM_FILES_TO_RUN" ] && [ "$NUM_FILES_TO_RUN" != "max" ]; then
        JSON_FILES=("${JSON_FILES[@]:0:$NUM_FILES_TO_RUN}")
    fi
    
    # Check if any JSON files exist on the line
    if [ ${#JSON_FILES[@]} -eq 0 ]; then
        echo "Warning: No JSON files found on line $line_num"
        continue
    fi
    
    echo "Found ${#JSON_FILES[@]} JSON files on line $line_num"
    
    # Create output directory if it doesn't exist
    mkdir -p "$LINE_OUTPUT_DIR"
    
    for json_file in "${JSON_FILES[@]}"; do
        # Extract experiment number from the filename
        # e.g., workspace/peoplejoin-qa/experiments/exp_configs_test/allergy_1/59.json -> 59
        FILENAME=$(basename "$json_file" .json)
        EXPERIMENT_NUM=$(echo "$FILENAME" | cut -d'_' -f3)
        
        # Check if this experiment has already been run
        LOG_FILE="$LINE_OUTPUT_DIR/${TASK_NAME}_${EXPERIMENT_NUM}.messages.json"
        
        if [ -f "$LOG_FILE" ]; then
            echo "Skipping Experiment $EXPERIMENT_NUM on line $line_num: already completed"
        else
            echo "----------------------------------------"
            echo "Experiment Set $line_num, Experiment $EXPERIMENT_NUM, Running $(basename "$json_file")"
            echo "----------------------------------------"
        
            # Run the experiment with the single JSON file with retries
            SUCCESS=false
            for i in {1..3}; do
                uv run python -m src.experimentation.experiment_with_hitl_or_simulation "$json_file" "$LINE_OUTPUT_DIR" --agent_config_path "$AGENT_CONFIG_PATH" --load_pth "$LINE_OUTPUT_DIR" --port $PORT
                if [ $? -eq 0 ]; then
                    SUCCESS=true
                    break
                fi
                echo "Attempt $i failed. Retrying in 5 seconds..."
                sleep 5
            done

            if $SUCCESS; then
                echo "✓ Experiment $EXPERIMENT_NUM on line $line_num completed successfully"
            else
                echo "✗ Experiment $EXPERIMENT_NUM on line $line_num failed after 3 attempts"
            fi
            echo ""

            # Make system prompt with retries, unless algo is baseline
            if [ "$ALG_NAME" = "baseline" ]; then
                echo "Skipping offline compute for baseline algorithm."
            else
                SUCCESS=false
                for i in {1..3}; do
                    uv run python -m src.experimentation.offline_compute --log_path "$LOG_FILE" --algo $ALG_NAME --history $HISTORY
                    if [ $? -eq 0 ]; then
                        SUCCESS=true
                        break
                    fi
                    echo "Offline compute attempt $i failed. Retrying in 5 seconds..."
                    sleep 5
                done

                if $SUCCESS; then
                    echo "✓ Offline compute for $LOG_FILE completed successfully"
                else
                    echo "✗ Offline compute for $LOG_FILE failed after 3 attempts"
                fi
                echo ""
            fi
        fi
    done

    # remove latest_logs.txt and latest_hindsight.txt from LINE_OUTPUT_DIR
    # prevents the wrong state from being accidentally read
    rm -f "$LINE_OUTPUT_DIR/latest_logs.txt"
    rm -f "$LINE_OUTPUT_DIR/latest_hindsight.txt"
done

echo "All experiments completed!"
