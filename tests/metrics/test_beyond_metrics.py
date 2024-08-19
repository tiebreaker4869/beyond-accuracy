from beyond_accuracy.metrics.beyond_metrics import Serendipity, cluster_based_serendipity
import torch


def test_cluster_based_serendipity():
    # 示例输入数据
    histories = [[1, 2], [3, 4], [5, 6]]
    recommendations = [[1, 3, 5], [2, 4, 6], [1, 2, 3]]
    item_embedding = torch.tensor([
        [0.1, 0.2],
        [0.2, 0.1],
        [0.3, 0.4],
        [0.4, 0.3],
        [0.5, 0.6],
        [0.6, 0.5],
        [0.7, 0.8],
    ])
    k = 2
    
    # 调用函数
    result = cluster_based_serendipity(histories, recommendations, item_embedding, k)

def test_unexpectedness_based_serendipity():
    # 示例输入数据
    histories = [[1, 2], [3, 4], [5, 6]]
    recommendations = [[1, 3, 5], [2, 4, 6], [1, 2, 3]]
    target_items = [[1, 2], [3, 4], [5, 6]]
    k = 2
    
    # 创建 Serendipity 实例
    serendipity_instance = Serendipity(histories, recommendations, target_items, k)
    
    # 调用 unexpectedness_based_serendipity 方法
    result = serendipity_instance.unexpectedness_based_serendipity()

def test_unexpected_based_orderaware_serendipity():
    # 示例输入数据
    histories = [[1, 2], [3, 4], [5, 6]]
    recommendations = [[1, 3, 5], [2, 4, 6], [1, 2, 3]]
    target_items = [[1, 2], [3, 4], [5, 6]]
    k = 2
    
    # 创建 Serendipity 实例
    serendipity_instance = Serendipity(histories, recommendations, target_items, k)
    
    # 调用 unexpected_based_orderaware_serendipity 方法
    result = serendipity_instance.unexpected_based_orderaware_serendipity()