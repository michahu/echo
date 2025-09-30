import json
import os
from pathlib import Path
import numpy as np
import argparse

def get_experiment_order(experiment_order_file):
    """Reads the experiment order file and returns a list of lists of experiment config file paths."""
    with open(experiment_order_file, 'r') as f:
        # Each line is a space-separated list of paths
        return [line.strip().split() for line in f.readlines()]

def reorder_metrics(metrics_file_path, experiment_order_base_path, row_index):
    """
    Reorders metrics from a metrics.json file based on the experiment order.
    The metrics.json can contain results from multiple experiments.
    We group by experiment, get the correct experiment_order.txt, and reorder.

    Args:
        metrics_file_path (str): The path to the metrics.json file.
        experiment_order_base_path (str): The base path to find experiment_order.txt files.
        row_index (int): The 0-based index of the row in experiment_order.txt to use.
    """
    metrics_file_path = Path(metrics_file_path)
    with open(metrics_file_path, 'r') as f:
        metrics_data = json.load(f)

    # Get the raw metrics data
    messages_all_raw = metrics_data.get('task_efficiency', {}).get('task_efficiency_messages_all_raw', [])
    overlap_raw = metrics_data.get('reference_overlap_using_llm_raw', {})

    # Group metrics by experiment name from overlap_raw keys
    # e.g., 'allergy_1_60' -> experiment 'allergy_1'
    grouped_metrics = {}
    for i, key in enumerate(overlap_raw.keys()):
        parts = key.split('_')
        exp_name = '_'.join(parts[:-1])
        exp_id = parts[-1]
        
        if exp_name not in grouped_metrics:
            grouped_metrics[exp_name] = []
        grouped_metrics[exp_name].append({'original_index': i, 'exp_id': exp_id, 'overlap_key': key})

    reordered_groups = {}

    for exp_name, items in grouped_metrics.items():
        experiment_order_file = Path(experiment_order_base_path) / exp_name / 'experiment_order.txt'
        if not experiment_order_file.exists():
            print(f"Experiment order file not found: {experiment_order_file}")
            continue
        
        experiment_order_list = get_experiment_order(experiment_order_file)
        try:
            experiment_order = experiment_order_list[row_index]
        except IndexError:
            print(f"Row index {row_index} is out of bounds for {experiment_order_file} with {len(experiment_order_list)} lines.")
            continue

        experiment_name_to_order_index = {Path(p).name.replace('.json', ''): i for i, p in enumerate(experiment_order)}

        def get_sort_key(item):
            return experiment_name_to_order_index.get(item['exp_id'], float('inf'))

        sorted_items = sorted(items, key=get_sort_key)

        # Create reordered arrays with zero imputation for missing values
        reordered_messages = []
        reordered_overlap = []
        
        # Create a mapping from exp_id to available data
        available_data = {item['exp_id']: item for item in sorted_items}
        
        # Process all expected experiments in order
        for exp_config_path in experiment_order:
            exp_id = Path(exp_config_path).name.replace('.json', '')
            if exp_id in available_data:
                item = available_data[exp_id]
                reordered_messages.append(messages_all_raw[item['original_index']])
                reordered_overlap.append(overlap_raw[item['overlap_key']])
            else:
                # Impute zero for missing values
                reordered_messages.append(0)
                reordered_overlap.append(0)

        reordered_groups[exp_name] = {
            'task_efficiency': reordered_messages,
            'overlap': reordered_overlap
        }

    return reordered_groups


def process_directory(base_dir, experiment_order_base_path):
    """
    Processes all metrics.json files in a directory, reorders them, and saves separate CSV files for each experiment.

    Args:
        base_dir (str): The base directory to search for metrics.json files.
        experiment_order_base_path (str): The base path to the experiment_order.txt files.
    """
    # Dictionary to group data by experiment name
    experiment_data = {}
    
    for metrics_file in Path(base_dir).rglob('metrics.json'):
        print(f"Processing {metrics_file}...")
        
        # Get the directory containing metrics.json as the key
        metrics_dir = str(metrics_file.parent)
        
        try:
            row_index = int(metrics_file.parts[-2]) - 1 # Get the row index from the folder name (e.g. '1' -> 0)
        except (IndexError, ValueError):
            print(f"Could not determine row index from path: {metrics_file}")
            continue

        reordered_groups = reorder_metrics(metrics_file, experiment_order_base_path, row_index)
        
        for exp_name, metrics in reordered_groups.items():
            if exp_name not in experiment_data:
                experiment_data[exp_name] = {
                    'task_efficiency': [],
                    'overlap': []
                }
            
            experiment_data[exp_name]['task_efficiency'].append([metrics_dir] + metrics['task_efficiency'])
            experiment_data[exp_name]['overlap'].append([metrics_dir] + metrics['overlap'])

    if not experiment_data:
        print("No data to process.")
        return

    # Save separate CSV files for each experiment
    base_path = Path(base_dir)
    for exp_name, data in experiment_data.items():
        # Save task efficiency CSV
        efficiency_file = base_path / f"{exp_name}_task_efficiency.csv"
        efficiency_array = np.array(data['task_efficiency'], dtype=object)
        np.savetxt(efficiency_file, efficiency_array, delimiter=",", fmt='%s')
        
        # Save overlap CSV
        overlap_file = base_path / f"{exp_name}_overlap.csv"
        overlap_array = np.array(data['overlap'], dtype=object)
        np.savetxt(overlap_file, overlap_array, delimiter=",", fmt='%s')
        
        print(f"Saved {exp_name} data to {efficiency_file} and {overlap_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collate metrics from experiment runs.")
    parser.add_argument("--base_dir", default="/home/azureuser/localfiles/multi-agent-collab/peoplejoin/workspace/serial/", help="The base directory to search for metrics.json files.")
    parser.add_argument("--order_dir", default="/home/azureuser/localfiles/multi-agent-collab/peoplejoin/workspace/peoplejoin-qa/experiments/exp_configs_test", help="The base path to the experiment_order.txt files.")

    args = parser.parse_args()

    process_directory(args.base_dir, args.order_dir)
