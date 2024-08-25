# beyond_accuracy

beyond_accuracy is a simple toolkit for evaluating recommender systems in terms of accuracy metrics and many metrics beyond them(such as serendipity, diversity, etc.)

## Installation
**!!!We highly recommend you to install the package in a separate virtual environment!!!**

```shell
$ git clone https://github.com/tiebreaker4869/beyond-accuracy.git
$ cd beyond-accuracy
$ python3 setup.py sdist
$ python3 -m pip install dist/beyond_accuracy-1.0.0.tar.gz
```

## Quick Start Examples

**Accuracy Metrics**
```python
from beyond_accuracy.metrics.acc_metrics import *

recommendations = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
target_items = [[1, 2], [5, 6], [8, 9]]
k = 2
assert hit_rate(recommendations, target_items, k) == 1

recommendations = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
target_items = [[1, 2], [4, 5], [7, 8]]
k = 2
assert ndcg(recommendations, target_items, k) == 1
```

**Serendipity Metrics**
```python
histories = [[1, 2, 6], [3, 4], [5, 6]]
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
    
result = cluster_based_serendipity(histories, recommendations, item_embedding, k)
```

## Documentation

**Currently Under Construction...**