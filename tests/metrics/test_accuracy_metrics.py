import pytest
import math
from beyond_accuracy.metrics.accuracy_metrics import Precision, Recall, MRR, NDCG


@pytest.fixture
def sample_data():
    recommended_list = [1, 2, 3, 4, 5]
    relevant_list = [1, 3, 6, 7]
    k = 3
    all_items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return recommended_list, relevant_list, k, all_items


def test_precision(sample_data):
    recommended_list, relevant_list, k, all_items = sample_data
    precision = Precision(all_items)
    assert math.isclose(
        precision.compute(recommended_list, relevant_list, [], k), 2 / 3
    )


def test_recall(sample_data):
    recommended_list, relevant_list, k, all_items = sample_data
    recall = Recall(all_items)
    assert math.isclose(recall.compute(recommended_list, relevant_list, [], k), 2 / 4)


def test_mrr(sample_data):
    recommended_list, relevant_list, k, all_items = sample_data
    mrr = MRR(all_items)
    assert math.isclose(mrr.compute(recommended_list, relevant_list, [], k), 1 / 1)


def test_ndcg(sample_data):
    recommended_list, relevant_list, k, all_items = sample_data
    ndcg = NDCG(all_items)
    golden_answer = (1 / 1 + 1 / 2) / (1 + 1 / math.log2(3) + 1 / 2)
    assert math.isclose(
        ndcg.compute(recommended_list, relevant_list, [], k), golden_answer
    )
