from src.async_collab.orchestrator.datum import AsyncCollabOutputDatum
from src.evaluation.eval import AsyncCollabDatumMessages, AsyncCollabMetric
from src.evaluation.eval_impls import (
    AsyncCollabFairnessMetric,
    AsyncCollabMetricEfficiency,
    AsyncCollabPeopleReferenceMetric,
    ChecklistMetric,
    InfoSeekingReferenceOverlapChecklist,
)


class AsyncCollabMetricManager:
    metrics: list[AsyncCollabMetric] = []

    def __init__(
        self, metric_names: list[str] | None = None, cache_file_name: str | None = None
    ) -> None:
        if metric_names is None:
            metric_names = ["task_efficiency"]
        for metric_name in metric_names:
            match metric_name:
                case "task_efficiency":
                    self.metrics.append(AsyncCollabMetricEfficiency())
                case "checklist":
                    self.metrics.append(ChecklistMetric())
                case "reference_overlap_using_llm":
                    self.metrics.append(
                        InfoSeekingReferenceOverlapChecklist(
                            cache_file_name=cache_file_name
                        )
                    )
                case AsyncCollabFairnessMetric.metric_name:
                    self.metrics.append(AsyncCollabFairnessMetric())
                case AsyncCollabPeopleReferenceMetric.metric_name:
                    self.metrics.append(AsyncCollabPeopleReferenceMetric())
                case _:
                    raise ValueError(f"Unknown metric name: {metric_name}")

    def __call__(self, prediction: AsyncCollabOutputDatum):
        prediction_messages = AsyncCollabDatumMessages.from_datum(prediction)
        # can use reference_messages later if needed
        for metric in self.metrics:
            metric(prediction, prediction_messages)

    def compute(self, attach_raw_score: bool = True) -> dict[str, float]:
        ret = {}
        for metric in self.metrics:
            ret[metric.metric_name] = metric.compute()
            if attach_raw_score:
                tmp = metric.get_raw_scores()
                if len(tmp) > 0:
                    ret[f"{metric.metric_name}_raw"] = tmp
        return ret

    def reset(self):
        for metric in self.metrics:
            metric.reset()
