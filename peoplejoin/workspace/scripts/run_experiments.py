#!/usr/bin/env python3
"""
Python script to run stateful experiments.
Replaces the PowerShell script logic with more robust error handling and logging.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def should_process_line(line_num: int, line_nums_spec: str | None) -> bool:
    """Check if a line number should be processed based on the specification."""
    if not line_nums_spec:
        return True

    # Handle comma-separated list and ranges
    line_specs = line_nums_spec.split(",")
    for spec in line_specs:
        spec = spec.strip()
        if "-" in spec:
            # Handle range (e.g., 1-3)
            try:
                start, end = map(int, spec.split("-"))
                if start <= line_num <= end:
                    return True
            except ValueError:
                print(f"Warning: Invalid range specification '{spec}'")
        else:
            # Handle single number
            try:
                if line_num == int(spec):
                    return True
            except ValueError:
                print(f"Warning: Invalid line number '{spec}'")

    return False


def run_experiment(
    json_file: str,
    output_dir: str,
    agent_config_path: str,
    port: str,
    max_retries: int = 3,
) -> bool:
    """Run a single experiment with retries."""
    print(f"Running experiment with {json_file}")

    for attempt in range(1, max_retries + 1):
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = "src"

            cmd = [
                sys.executable,
                "-m",
                "src.experimentation.experiment_with_hitl_or_simulation",
                json_file,
                output_dir,
                "--agent_config_path",
                agent_config_path,
                "--load_pth",
                output_dir,
                "--port",
                port,
                "--use_substrate",
            ]

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                print("✓ Experiment completed successfully")
                return True
            else:
                print(f"Experiment failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error output: {result.stderr}")

        except Exception as e:
            print(f"Exception occurred: {e}")

        if attempt < max_retries:
            print(f"Attempt {attempt} failed. Retrying in 5 seconds...")
            time.sleep(5)

    print(f"✗ Experiment failed after {max_retries} attempts")
    return False


def run_offline_compute(
    log_file: str, algorithm: str, history: str, max_retries: int = 3
) -> bool:
    """Run offline compute with retries."""
    if algorithm == "baseline":
        print("Skipping offline compute for baseline algorithm.")
        return True

    print(f"Running offline compute for {log_file}")

    for attempt in range(1, max_retries + 1):
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = "src"

            cmd = [
                sys.executable,
                "-m",
                "src.experimentation.offline_compute",
                "--log_path",
                log_file,
                "--algo",
                algorithm,
                "--history",
                history,
            ]

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                print("✓ Offline compute completed successfully")
                return True
            else:
                print(f"Offline compute failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error output: {result.stderr}")

        except Exception as e:
            print(f"Exception occurred: {e}")

        if attempt < max_retries:
            print(f"Offline compute attempt {attempt} failed. Retrying in 5 seconds...")
            time.sleep(5)

    print(f"✗ Offline compute failed after {max_retries} attempts")
    return False


def cleanup_temp_files(output_dir: str) -> None:
    """Remove temporary files from output directory."""
    temp_files = ["latest_logs.txt", "latest_hindsight.txt"]

    for temp_file in temp_files:
        temp_path = Path(output_dir) / temp_file
        if temp_path.exists():
            try:
                temp_path.unlink()
                print(f"Removed {temp_path}")
            except Exception as e:
                print(f"Warning: Could not remove {temp_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run stateful experiments")
    parser.add_argument(
        "num_files_to_run", help='Number of files to run per line (or "max")'
    )
    parser.add_argument("port", help="Port number")
    parser.add_argument("algorithm", help="Algorithm name")
    parser.add_argument("task_name", help="Task name")
    parser.add_argument("history", help="History setting")
    parser.add_argument(
        "line_nums", nargs="?", help="Line numbers to process (optional)"
    )
    parser.add_argument(
        "extra_naming", nargs="?", default="0", help="Extra naming parameter"
    )

    args = parser.parse_args()

    # Set up paths
    experiment_order_file = f"workspace/peoplejoin-qa/experiments/exp_configs_test/{args.task_name}/experiment_order.txt"
    output_dir = f"workspace/stateful-no-history-2/{args.algorithm}/{args.history}/{args.extra_naming}"
    agent_config_path = f"workspace/peoplejoin-qa/experiments/agent_configs/agentconf_{args.task_name}_devphi4_oneexample.json"

    # Check if experiment order file exists
    if not Path(experiment_order_file).exists():
        print(f"Error: File {experiment_order_file} does not exist")
        sys.exit(1)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Read experiment order file
    try:
        with open(experiment_order_file) as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {experiment_order_file}: {e}")
        sys.exit(1)

    # Show which lines will be processed
    if args.line_nums:
        print(f"Will process only lines: {args.line_nums}")
    else:
        print(f"Will process all {len(lines)} lines")

    # Process each line
    for i, line in enumerate(lines):
        line_num = i + 1
        line = line.strip()

        # Check if this line should be processed
        if not should_process_line(line_num, args.line_nums):
            print(
                f"Skipping line {line_num} (not in specified LINE_NUMS: {args.line_nums})"
            )
            continue

        line_output_dir = f"{output_dir}/{line_num}"
        print(f"Processing line {line_num}: {line}")

        # Parse JSON files from the line
        json_files = [f for f in line.split() if f.strip()]

        # Limit number of files if specified
        if args.num_files_to_run != "max" and args.num_files_to_run.isdigit():
            num_files = int(args.num_files_to_run)
            json_files = json_files[:num_files]

        if not json_files:
            print(f"Warning: No JSON files found on line {line_num}")
            continue

        print(f"Found {len(json_files)} JSON files on line {line_num}")

        # Create line output directory
        Path(line_output_dir).mkdir(parents=True, exist_ok=True)

        # Process each JSON file
        for json_file in json_files:
            # Extract experiment number from filename
            filename = Path(json_file).stem
            try:
                experiment_num = filename.split("_")[2]
            except IndexError:
                print(f"Warning: Could not extract experiment number from {filename}")
                experiment_num = filename

            # Check if experiment already completed
            log_file = (
                f"{line_output_dir}/{args.task_name}_{experiment_num}.messages.json"
            )

            if Path(log_file).exists():
                print(
                    f"Skipping Experiment {experiment_num} on line {line_num}: already completed"
                )
                continue

            print("=" * 50)
            print(
                f"Experiment Set {line_num}, Experiment {experiment_num}, Running {Path(json_file).name}"
            )
            print("=" * 50)

            # Run the experiment
            if run_experiment(json_file, line_output_dir, agent_config_path, args.port):
                # Run offline compute if experiment succeeded
                run_offline_compute(log_file, args.algorithm, args.history)

            print()  # Add blank line between experiments

        # Clean up temporary files
        cleanup_temp_files(line_output_dir)

    print("All experiments completed!")


if __name__ == "__main__":
    main()
