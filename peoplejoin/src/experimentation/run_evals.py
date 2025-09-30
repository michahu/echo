import json
from pathlib import Path

from src.async_collab.orchestrator.datum import AsyncCollabOutputDatum
from src.evaluation.eval_manager import AsyncCollabMetricManager

from tqdm import tqdm


def load_data_from_files(file_paths: list[Path]) -> list[AsyncCollabOutputDatum]:
    ret = []
    for file_path in file_paths:
        with open(file_path) as f:
            json_str = json.loads(f.read())
        datum = AsyncCollabOutputDatum.from_dict(json_str)
        # datum = jsons.loads(json_str, cls=AsyncCollabOutputDatum)
        ret.append(datum)
    return ret


def compute_evals(
    data: list[AsyncCollabOutputDatum],
    metric_names: list[str],
    cache_file_name: str | None = None,
) -> dict[str, float]:
    evals_runner: AsyncCollabMetricManager = AsyncCollabMetricManager(
        metric_names=metric_names, cache_file_name=cache_file_name
    )
    for datum in tqdm(data):
        evals_runner(datum)
    metrics = evals_runner.compute()
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_outputs_folder", type=str)
    parser.add_argument("--metrics_file", type=str)
    parser.add_argument(
        "--metric_names", type=str, nargs="*", default=["task_efficiency", "checklist"]
    )
    # cache_file_name
    parser.add_argument("--cache_file_name", type=str, default=None)
    args = parser.parse_args()

    saved_outputs_folder = Path(args.saved_outputs_folder)
    data = load_data_from_files(list(saved_outputs_folder.glob("*datum.json")))
    metric_names = args.metric_names
    metrics_file = Path(args.metrics_file)
    cache_file_name = args.cache_file_name

    metrics = compute_evals(data, metric_names, cache_file_name)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()

# python src/experimentation/run_evals.py workspace/sample_run/saved_output/ workspace/sample_run/metrics.json
# python src/experimentation/run_evals.py workspace/04-02-2024-end-to-end-eval/saved_outputs/ workspace/04-02-2024-end-to-end-eval/metrics.json task_efficiency,checklist,reference_overlap_using_llm,people_reference
