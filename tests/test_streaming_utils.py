import pytest
import queue
import time
import os
from zipmap.utils.streaming_utils import BaseWorker

def test_worker_lifecycle():
    q = queue.Queue()
    results = []
    class MockWorker(BaseWorker):
        def process(self, item):
            results.append(item)
    
    worker = MockWorker(q)
    worker.start()
    q.put(1)
    time.sleep(0.1)
    worker.stop()
    assert 1 in results

def test_poison_pill_hang():
    q = queue.Queue()
    class MockWorker(BaseWorker):
        def process(self, item):
            pass
    
    worker = MockWorker(q)
    worker.start()
    q.put(None)
    # This might hang if task_done is not called for None
    q.join() 
    worker.join()
    assert True

def test_result_saver_worker(tmp_path):
    import torch
    import numpy as np
    from zipmap.utils.streaming_utils import ResultSaverWorker
    
    q = queue.Queue()
    worker = ResultSaverWorker(q)
    worker.start()
    
    save_path = str(tmp_path / "test_results.npz")
    predictions = {
        "tensor": torch.tensor([1.0, 2.0, 3.0]),
        "array": np.array([4, 5, 6]),
        "scalar": 7.0
    }
    
    q.put({"predictions": predictions, "save_path": save_path})
    q.put(None)
    worker.join()
    
    assert os.path.exists(save_path)
    data = np.load(save_path)
    assert np.allclose(data["tensor"], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(data["array"], np.array([4, 5, 6]))
    assert data["scalar"] == 7.0
