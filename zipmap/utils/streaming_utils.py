import threading
import queue
import abc
import numpy as np
import os
import torch
from zipmap.utils.load_fn import load_and_preprocess_images

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
                if item is None:
                    self.in_queue.task_done()
                    break
                
                try:
                    result = self.process(item)
                    if self.out_queue and result is not None:
                        self.out_queue.put(result)
                except Exception as e:
                    print(f"Error in {self.name}: {e}")
                finally:
                    self.in_queue.task_done()
            except queue.Empty:
                continue

    @abc.abstractmethod
    def process(self, item):
        pass

    def stop(self):
        self.stop_event.set()
        self.join()

class ImageLoaderWorker(BaseWorker):
    def process(self, image_path):
        # 预处理图片
        # 注意: load_and_preprocess_images 接受一个列表，返回 (N, 3, H, W) 的 tensor
        tensor = load_and_preprocess_images([image_path])
        return {"path": image_path, "tensor": tensor}

class ResultSaverWorker(BaseWorker):
    def process(self, data):
        predictions = data["predictions"]
        save_path = data["save_path"]
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 使用 np.savez_compressed 以节省磁盘空间
        np.savez_compressed(save_path, **predictions)
        return None
