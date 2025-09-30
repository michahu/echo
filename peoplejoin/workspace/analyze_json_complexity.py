#!/usr/bin/env python3
"""
Script to analyze JSON files and calculate complexity based on user_id_to_descriptions count.
Outputs results to a CSV file.
"""

import json
import os
import csv
import argparse
from pathlib import Path


def calculate_complexity(json_file_path):
    """
    Calculate complexity of a JSON file based on user_id_to_descriptions count.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        int: Number of items in user_id_to_descriptions, or 0 if not found
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # Get the user_id_to_descriptions field
        user_descriptions = data.get('user_id_to_descriptions', {})
        return len(user_descriptions)
        
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Error processing {json_file_path}: {e}")
        return 0


def analyze_directory(directory_path, output_csv_path):
    """
    Analyze all JSON files in a directory and write results to CSV.
    
    Args:
        directory_path (str): Path to directory containing JSON files
        output_csv_path (str): Path for output CSV file
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory {directory_path} does not exist.")
        return
    
    # Find all JSON files in the directory
    json_files = list(directory.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return
    
    results = []
    
    # Process each JSON file
    for json_file in json_files:
        complexity = calculate_complexity(json_file)
        results.append({
            'json_name': json_file.name,
            'complexity': complexity
        })
        print(f"Processed {json_file.name}: complexity = {complexity}")
    
    # Sort results by complexity (descending) for better readability
    results.sort(key=lambda x: x['complexity'], reverse=True)
    
    # Write results to CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['json_name', 'complexity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults written to {output_csv_path}")
    print(f"Total files processed: {len(results)}")
    
    # Print summary statistics
    complexities = [r['complexity'] for r in results]
    if complexities:
        print(f"Average complexity: {sum(complexities) / len(complexities):.2f}")
        print(f"Max complexity: {max(complexities)}")
        print(f"Min complexity: {min(complexities)}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze JSON files and calculate complexity based on user_id_to_descriptions count'
    )
    parser.add_argument(
        'directory', 
        help='Directory containing JSON files to analyze'
    )
    parser.add_argument(
        '-o', '--output', 
        default='json_complexity_analysis.csv',
        help='Output CSV file path (default: json_complexity_analysis.csv)'
    )
    
    args = parser.parse_args()
    
    analyze_directory(args.directory, args.output)


if __name__ == '__main__':
    main()