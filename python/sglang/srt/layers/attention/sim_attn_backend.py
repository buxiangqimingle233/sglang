from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

class SimAttnBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        pass 
    
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass