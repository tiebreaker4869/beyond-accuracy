import pytest
import math

from beyond_accuracy.metrics.beyond_metrics import Serendipity, OrderAwareSerendipity
import random


@pytest.fixture
def sample_data():

    # 所有项目（假设有 100 个项目）
    all_items = list(range(1, 101))

    # 假设的所有交互数据（生成 500 个交互数据点）
    all_interactions = [random.choice(all_items) for _ in range(500)]

    # 推荐的项目（假设推荐了 20 个项目）
    recommendations = random.sample(all_items, 20)

    # 交互历史（假设用户有 50 个交互历史）
    interaction_history = random.sample(all_interactions, 50)
    k = 10
    return all_items, all_interactions, recommendations, interaction_history, k


def test_serendipity(sample_data):
    all_items, all_interactions, recommendations, interaction_history, k = sample_data
    serendipity_metric = Serendipity(all_items)
    serendipity_metric.fit(all_interactions)
    serendipity_value = serendipity_metric.compute(
        recommendations, interaction_history, k
    )
    print(serendipity_metric.get_itemwise_metrics())
    print(serendipity_value)


def test_serendipity_r(sample_data):
    all_items, all_interactions, recommendations, interaction_history, k = sample_data
    serendipity_metric = OrderAwareSerendipity(all_items)
    serendipity_metric.fit(all_interactions)
    serendipity_value = serendipity_metric.compute(
        recommendations, interaction_history, k
    )
    print(serendipity_value)
