import pytest
import queue
import time
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
