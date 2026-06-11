from tools.numba_gpu.core.process import BasePipeline

class Pipeline(BasePipeline):
    def __init__(self, config):
        
        super().__init__(config)
        
        self.threadsperblock = (128, 8, 1)