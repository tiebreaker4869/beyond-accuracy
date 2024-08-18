from beyond_accuracy.metrics.acc_metrics import hit_rate, ndcg, precision, recall

def test_hit_rate_full_hit():
    recommendations = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    target_items = [[1, 2], [5, 6], [8, 9]]
    k = 2
    assert hit_rate(recommendations, target_items, k) == 1
    
def test_hit_rate_partial_hit():
    recommendations = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    target_items = [[3, 4],  [6, 7], [8, 9]]
    k = 2
    assert hit_rate(recommendations, target_items, k) == 1 / 3
    
def test_hit_rate_no_hit():
    recommendations = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    target_items = [[10, 11],  [12, 13], [14, 15]]
    k = 2
    assert hit_rate(recommendations, target_items, k) == 0
    
def test_ndcg():
    recommendations = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    target_items = [[1, 2], [4, 5], [7, 8]]
    k = 2
    assert ndcg(recommendations, target_items, k) == 1

def test_precision():
    recommendations = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    target_items = [[1, 2], [5, 6], [8, 9]]
    k = 2
    assert precision(recommendations, target_items, k) == 2 / 3
    
def test_recall():
    recommendations = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    target_items = [[1, 2], [5, 6], [8, 9]]
    k = 2
    assert recall(recommendations, target_items, k) == 2 / 3