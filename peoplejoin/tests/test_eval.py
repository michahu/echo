import json

import jsons
from pytest import approx

from async_collab.orchestrator.datum import AsyncCollabOutputDatum
from evaluation.eval_impls import (
    AsyncCollabFairnessMetric,
    AsyncCollabMetricEfficiency,
    AsyncCollabPeopleReferenceMetric,
    count_message_tokens,
)


def get_test_datum():
    with open("tests/test_data/test_saved_datum.json") as f:
        json_str = f.read()
    json_dict = jsons.loads(json_str)
    return AsyncCollabOutputDatum.from_dict(json_dict)


def get_another_test_datum():
    with open("tests/test_data/test_saved_another_datum.json") as f:
        json_str = f.read()
    json_dict = jsons.loads(json_str)
    return AsyncCollabOutputDatum.from_dict(json_dict)


def test_count_message_tokens():
    assert count_message_tokens("hello world") == 2
    assert count_message_tokens("hello world how are you doing") == 6
    assert count_message_tokens("hello world how are you doing today") == 7
    assert count_message_tokens("hello world how are you doing today.") == 7


def test_async_collab_efficiency_metric():
    datum = get_test_datum()
    metric = AsyncCollabMetricEfficiency()
    metric(datum)
    metrics = metric.compute()
    print("metrics = ", json.dumps(metrics, indent=2))
    assert metrics["task_efficiency_message_count_from_primary_to_bot"] == approx(4.0)
    assert metrics["task_efficiency_message_count_from_bot_to_primary"] == approx(4.0)
    assert metrics["task_efficiency_message_count_from_secondary_to_bot"] == approx(3.0)
    assert metrics["task_efficiency_message_count_from_bot_to_secondary"] == approx(3.0)
    assert metrics["task_efficiency_messages_all"] == approx(14.0)
    assert metrics["task_efficiency_total_task_time"] == approx(28843.84)
    assert metrics["task_efficiency_messages_size_from_primary_to_bot"] == approx(22.0)
    assert metrics["task_efficiency_messages_size_from_bot_to_primary"] == approx(81.0)
    assert metrics["task_efficiency_messages_size_from_secondary_to_bot"] == approx(
        94.0
    )
    assert metrics["task_efficiency_messages_size_from_bot_to_secondary"] == approx(
        41.0
    )
    assert metrics["task_efficiency_people_contacted"] == approx(2.0)
    assert metrics["task_efficiency_count"] == approx(1)


def test_fairness_metric():
    datum = get_test_datum()
    datum2 = get_another_test_datum()
    metric = AsyncCollabFairnessMetric()
    metric(datum)
    metric(datum2)
    metrics = metric.compute()
    print("metrics = ", json.dumps(metrics, indent=2))
    assert metrics["fairness_max_message_count_for_any_user"] == approx(4.0)
    assert metrics["fairness_max_message_normalized_count_for_any_user"] == approx(0.67)
    assert metrics["fairness_variance_across_users"] == approx(0.0556)
    assert metrics["fairness_count"] == approx(2)
    metrics_raw = metric.get_raw_scores()
    print("metrics_raw = ", json.dumps(metrics_raw, indent=2))
    assert metrics_raw["fairness_message_count_to_user_bhushan"] == approx(2.0)
    assert metrics_raw["fairness_message_count_to_user_ruwaidah"] == approx(4.0)


def test_people_contacted_metric():
    datum = get_test_datum()
    assert datum.metadata.description_reference_people == ["ruwaidah", "bhushan"]
    metric = AsyncCollabPeopleReferenceMetric()
    metric(datum)
    metrics = metric.compute()
    assert metrics["people_contacted_precision"] == approx(1.0)
    assert metrics["people_contacted_recall"] == approx(1.0)
    metric.reset()
    datum = get_another_test_datum()
    assert datum.metadata.description_reference_people == [
        "irena",
        "ruwaidah",
        "bhushan",
    ]
    metric(datum)
    metrics = metric.compute()
    assert metrics["people_contacted_precision"] == approx(1.0)
    assert metrics["people_contacted_recall"] == approx(0.67)


# python -m pytest tests/test_eval.py::test_async_collab_efficiency_metric -ss
# python -m pytest tests/test_eval.py::test_fairness_metric -ss
# python -m pytest tests/test_eval.py::test_people_contacted_metric -ss
