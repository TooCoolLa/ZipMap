import threading
import queue
import abc

class BaseWorker(threading.Thread, abc.ABC):
    def __init__(self, in_queue, out_queue=None, name=None):
        super().__init__(name=name)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set() or not self.in_queue.empty():
            try:
                item = self.in_queue.get(timeout=0.1)
                if item is None: break
                result = self.process(item)
                if self.out_queue and result is not None:
                    self.out_queue.put(result)
                self.in_queue.task_done()
            except queue.Empty:
                continue

    @abc.abstractmethod
    def process(self, item):
        pass

    def stop(self):
        self.stop_event.set()
        self.join()
