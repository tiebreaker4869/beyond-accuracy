import pytest
import math

from beyond_accuracy.metrics.beyond_metrics import Serendipity, OrderAwareSerendipity


@pytest.fixture
def sample_data():
    all_items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # 5, 4, 3, 2, 1, 1, 1, 1, 1, 1
    all_interactions = [1, 2, 3, 4, 1, 2, 3, 1, 2, 1]  # 假设的所有交互数据
    recommendations = [1, 2, 3, 4, 5]
    interaction_history = [1, 3, 6, 7]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    k = 3
    return all_items, all_interactions, recommendations, interaction_history, scores, k


def test_serendipity(sample_data):
    all_items, all_interactions, recommendations, interaction_history, scores, k = (
        sample_data
    )
    serendipity_metric = Serendipity(all_items)
    serendipity_metric.fit(all_interactions)
    serendipity_value = serendipity_metric.compute(
        recommendations, interaction_history, scores, k
    )
    popularity_based_scores = {
        1: 5 / 20,
        2: 4 / 20,
        3: 3 / 20,
        4: 2 / 20,
        5: 1 / 20,
        6: 1 / 20,
        7: 1 / 20,
        8: 1 / 20,
        9: 1 / 20,
        10: 1 / 20,
    }
    # 计算预期的 serendipity 值
    expected_serendipity = 0.0
    for i, item in enumerate(recommendations[:k]):
        if item in interaction_history:
            expected_serendipity += max((scores[i] - popularity_based_scores[item]), 0)
    expected_serendipity /= k

    assert math.isclose(serendipity_value, expected_serendipity)


def test_serendipity_r(sample_data):
    all_items, all_interactions, recommendations, interaction_history, scores, k = (
        sample_data
    )
    serendipity_metric = OrderAwareSerendipity(all_items)
    serendipity_metric.fit(all_interactions)
    serendipity_value = serendipity_metric.compute(
        recommendations, interaction_history, scores, k
    )
    popularity_based_scores = {
        1: 5 / 20,
        2: 4 / 20,
        3: 3 / 20,
        4: 2 / 20,
        5: 1 / 20,
        6: 1 / 20,
        7: 1 / 20,
        8: 1 / 20,
        9: 1 / 20,
        10: 1 / 20,
    }
    # 计算预期的 serendipity 值
    expected_serendipity = 0.0
    relevant_cnt = 0
    for i, item in enumerate(recommendations[:k]):
        if item in interaction_history:
            relevant_cnt += 1
            expected_serendipity += max((scores[i] - popularity_based_scores[item]), 0) * (relevant_cnt / (i + 1))
    expected_serendipity /= k

    assert math.isclose(serendipity_value, expected_serendipity)