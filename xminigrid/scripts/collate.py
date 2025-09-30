#!/usr/bin/env python3
"""
Collate results from XMiniGrid experiments into a CSV file.
This script traverses the results directory structure and extracts:
- method_name: The method used (e.g., awm, gce, none, reflexion)
- environment: The environment number (0-5)
- seed: Extracted from results_{i}.json filename
- reward: The reward value from the JSON file
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def extract_seed_from_filename(filename: str) -> int:
    """Extract seed number from results_{seed}.json filename."""
    match = re.search(r"results_(\d+)\.json", filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract seed from filename: {filename}")


def load_json_file(filepath: Path) -> dict[str, Any]:
    """Load and parse a JSON file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading {filepath}: {e}")
        return {}


def collect_results(results_dir: Path) -> list[dict[str, Any]]:
    """
    Traverse the results directory and collect all experiment results.

    Expected directory structure:
    results/
        model_name/
            method_name/
                environment_number/
                    results_{seed}.json
    """
    data = []

    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return data

    # Traverse the directory structure
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        print(f"Processing model: {model_name}")

        for method_dir in model_dir.iterdir():
            if not method_dir.is_dir():
                continue

            method_name = method_dir.name
            print(f"  Processing method: {method_name}")

            for env_dir in method_dir.iterdir():
                if not env_dir.is_dir():
                    continue

                try:
                    environment = int(env_dir.name)
                except ValueError:
                    print(
                        f"    Skipping non-numeric environment directory: {env_dir.name}"
                    )
                    continue

                print(f"    Processing environment: {environment}")

                # Look for results_*.json files
                for json_file in env_dir.glob("results_*.json"):
                    try:
                        seed = extract_seed_from_filename(json_file.name)
                        json_data = load_json_file(json_file)

                        if "reward" in json_data:
                            row = {
                                "model_name": model_name,
                                "method_name": method_name,
                                "environment": environment,
                                "seed": seed,
                                "reward": json_data["reward"],
                            }
                            data.append(row)
                            print(
                                f"      Added: {json_file.name} (reward: {json_data['reward']})"
                            )
                        else:
                            print(f"      Warning: No 'reward' field in {json_file}")

                    except Exception as e:
                        print(f"      Error processing {json_file}: {e}")

    return data


def write_csv(data: list[dict[str, Any]], output_file: Path) -> None:
    """Write the collected data to a CSV file."""
    if not data:
        print("No data to write to CSV")
        return

    fieldnames = ["model_name", "method_name", "environment", "seed", "reward"]

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Successfully wrote {len(data)} rows to {output_file}")


def main():
    """Main function to orchestrate the collation process."""
    parser = argparse.ArgumentParser(
        description="Collate results from XMiniGrid experiments into a CSV file."
    )

    # Define paths relative to the script location for default
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    default_results_dir = project_root / "results"

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help=f"Path to the results directory (default: {default_results_dir})",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "collated_results.csv",
        help=f"Output CSV file path (default: {project_root / 'collated_results.csv'})",
    )

    args = parser.parse_args()

    results_dir = args.results_dir
    output_file = args.output

    print(f"Looking for results in: {results_dir}")
    print(f"Output will be written to: {output_file}")

    # Collect all results
    data = collect_results(results_dir)

    if data:
        # Sort data for consistent output
        data.sort(
            key=lambda x: (
                x["model_name"],
                x["method_name"],
                x["environment"],
                x["seed"],
            )
        )

        # Write to CSV
        write_csv(data, output_file)

        # Print summary statistics
        print("\nSummary:")
        print(f"Total experiments: {len(data)}")

        models = {row["model_name"] for row in data}
        print(f"Models found: {sorted(models)}")

        methods = {row["method_name"] for row in data}
        print(f"Methods found: {sorted(methods)}")

        environments = {row["environment"] for row in data}
        print(f"Environments found: {sorted(environments)}")

        seeds = {row["seed"] for row in data}
        print(f"Seeds found: {sorted(seeds)}")

        avg_reward = sum(row["reward"] for row in data) / len(data)
        print(f"Average reward: {avg_reward:.4f}")

    else:
        print("No valid results found to collate.")


if __name__ == "__main__":
    main()
