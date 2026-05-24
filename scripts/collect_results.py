"""
Collect and aggregate results from stage-2 training runs.
Compares KNN baseline vs adaptive across precision, recall, F1, AUC-PR.
Outputs comparison.csv and comparison.json.
"""

import json
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

def collect_results(runs_dir='runs', output_dir='outputs'):
    """
    Scan runs_dir for run111_knn and run111_adaptive directories.
    Extract metrics from metrics_*.json files.
    Aggregate and save to CSV + JSON.
    """

    runs_path = Path(runs_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Find KNN and adaptive runs
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name

        if 'run111_knn' in run_name:
            mode = 'knn'
        elif 'run111_adaptive' in run_name:
            mode = 'adaptive'
        else:
            continue

        metrics_dir = run_dir / 'images'
        if not metrics_dir.exists():
            print(f"Skipping {run_name}: no images/ directory")
            continue

        # Load metrics
        metrics_files = list(metrics_dir.glob('metrics_*.json'))
        if not metrics_files:
            print(f"Skipping {run_name}: no metrics_*.json files")
            continue

        print(f"Found {run_name}")
        mode_results = {
            'run_name': run_name,
            'metrics_files': []
        }

        for metrics_file in metrics_files:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            mode_results['metrics_files'].append({
                'file': metrics_file.name,
                'metrics': metrics
            })

        results[mode] = mode_results

    # Extract key metrics
    comparison = {
        'timestamp': str(Path(output_path).stat()),
        'runs_compared': list(results.keys()),
        'metrics': {}
    }

    for mode, data in results.items():
        print(f"\n{mode.upper()}:")
        print(f"  Run: {data['run_name']}")

        metrics_data = {}
        for mf in data['metrics_files']:
            metrics_dict = mf['metrics']
            print(f"  From {mf['file']}:")

            # Try to extract common metrics
            for key in ['precision', 'recall', 'f1', 'f1_macro', 'f1_micro', 'auc_pr']:
                if key in metrics_dict:
                    metrics_data[key] = metrics_dict[key]
                    print(f"    {key}: {metrics_dict[key]}")

            # Look for edge-specific metrics
            if 'edges' in metrics_dict:
                for edge_key, edge_val in metrics_dict['edges'].items():
                    metrics_data[f'edges_{edge_key}'] = edge_val
                    print(f"    edges.{edge_key}: {edge_val}")

        comparison['metrics'][mode] = metrics_data

    # Write JSON
    json_path = output_path / 'comparison.json'
    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved to {json_path}")

    # Write CSV
    csv_path = output_path / 'comparison.csv'
    if comparison['metrics']:
        all_keys = set()
        for mode_metrics in comparison['metrics'].values():
            all_keys.update(mode_metrics.keys())
        all_keys = sorted(list(all_keys))

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['mode'] + all_keys)
            for mode in sorted(comparison['metrics'].keys()):
                row = [mode]
                metrics_dict = comparison['metrics'][mode]
                for key in all_keys:
                    row.append(metrics_dict.get(key, ''))
                writer.writerow(row)

        print(f"Saved to {csv_path}")

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    for mode, metrics in comparison['metrics'].items():
        print(f"\n{mode.upper()}:")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")

if __name__ == '__main__':
    collect_results()
