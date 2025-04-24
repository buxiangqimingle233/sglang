""" 
This file contains two classes: `ModelTracer` and `ModelRunnerSim`. 

- ModelTracer is used to trace the output of each forward batch and save trace file.
- ModelRunnerSim replaces the ModelRunner to return the output of each forward batch. 
Instead of actually invoking the forward backend, it uses the trace file to retrive the output and keep the functionality of sglang.
This enables us to decouple the functionality and the performance, thereby enabling evaluating sglang on arbitrary hardware backends.  
"""

import os
import atexit
import json
import logging
from collections import defaultdict
import pickle
from turtle import forward
import torch
import time

from typing import List, Optional, Tuple, Union, Dict, Any
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.layers.sampler import Sampler
from sglang.srt.layers.attention.sim_attn_backend import SimAttnBackend
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.paged_allocator import PagedTokenToKVPoolAllocator
from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolSim, TokenToKVPoolAllocator
from sglang.srt.layers.dp_attention import get_attention_tp_size


from sglang.srt.utils import (
    MultiprocessingSerializer,
    enable_show_time_cost,
    init_custom_process_group,
    is_cuda,
    is_hip,
    monkey_patch_p2p_access_check,
    monkey_patch_vllm_gguf_config,
    set_cpu_offload_max_bytes,
    set_cuda_arch,
)

import sglang.srt.utils as utils


logger = logging.getLogger(__name__)

SGLANG_CI_SMALL_KV_SIZE = os.getenv("SGLANG_CI_SMALL_KV_SIZE", None)


class ForwardResultTracer:
    """
    A tracer module for capturing model outputs.
    
    This class is responsible for tracking outputs from model execution and persisting
    them to disk. It operates independently from the model runner to avoid modifying
    the original execution flow.
    """
    
    trace_file_path = "trace/model_trace__0.jsonl"
    
    def __init__(self, enable_execution_tracing: bool = False, tp_rank: int = 0):
        self.should_trace = enable_execution_tracing and tp_rank == 0  # Only log on rank 0 to avoid duplication
        self.tp_rank = tp_rank

        self._trace_cache = defaultdict(list)  # Using request ID -> [output_ids]
        self._finished_requests = set()  # Track completed request IDs

        self.logger = logger

        if self.should_trace:
            self.logger.info(f"ModelTracer initialized for tp_rank: {tp_rank}")
            self._initialize_trace_file()

    def trace(self, 
              next_token_ids: Optional[torch.Tensor] = None, 
              schedule_batch: Optional[ScheduleBatch] = None) -> None:
        """
        Trace the output of a forward pass. Only writes to the cache.
        
        Args:
            next_token_ids: The sampled token IDs (optional)
            schedule_batch: The batch information including request IDs and output locations
        """
        if not self.should_trace or schedule_batch is None or next_token_ids is None: 
            return
        try:
            # Convert tensor to list for serialization
            next_token_ids = next_token_ids.detach().cpu().tolist() if hasattr(next_token_ids, 'detach') else next_token_ids.tolist()

            # Add next_token_ids to the cache, indexed by request ID
            for req, next_token_id in zip(schedule_batch.reqs, next_token_ids):
                self._trace_cache[req.rid].append(next_token_id)
                # Mark as completed if the request is finished
                if req.finished():
                    self._mark_request_completed(req.rid)
        except Exception as e:
            self.logger.error(f"Error tracing model output: {e}")


    # TODO: the ForwardResultTracer executes at the Scheduler's subprocess, which would be killed by main process 
    # via srt/utils.py:kill_process_tree when it terminates. This disables us to register an elegant cache flushing
    # procedure at the process exit time. To avoid bugs, we need to dump the trace at each batch ends. It may introduce
    # overheads due to the frequent file writes. 
    def check_and_flush_finished_requests(self, schedule_batch: ScheduleBatch) -> None:
        if not self.should_trace:
            return
        try:
            for req in schedule_batch.reqs:
                if req.finished(): 
                    self._mark_request_completed(req.rid)
            self._flush_finished_requests()
        except Exception as e:
            self.logger.error(f"Error checking and flushing requests: {e}")

    def _initialize_trace_file(self):
        """Create the trace file with timestamp at first call."""
        # Create trace directory if it doesn't exist
        if not os.path.exists(os.path.dirname(self.trace_file_path)):
            os.makedirs(os.path.dirname(self.trace_file_path), exist_ok=True)
        open(self.trace_file_path, "w").close()                      # Create an empty file
        self.logger.info(f"Trace stored to {self.trace_file_path}")
    
    def _mark_request_completed(self, request_id: str) -> None:
        """Mark a request as completed so it can be flushed to disk."""
        if request_id in self._trace_cache and request_id not in self._finished_requests:
            self._finished_requests.add(request_id)
            self.logger.info(f"Request {request_id} marked as completed")

    def _flush_finished_requests(self) -> None:
        if not self._finished_requests or not self.trace_file_path:
            return
        
        requests_to_flush = self._finished_requests.intersection(self._trace_cache.keys())
        if not requests_to_flush:
            return
        self.logger.info(f"Flushing {len(requests_to_flush)} completed requests to {self.trace_file_path}")

        try:
            with open(self.trace_file_path, 'a') as f:
                for rid in requests_to_flush:
                    f.write(json.dumps({rid: self._trace_cache[rid]}) + "\n")
                    # Remove from cache after flushing
                    self._trace_cache.pop(rid)
                    self._finished_requests.remove(rid)
        except Exception as e:
            self.logger.error(f"Error writing trace file: {e}")

    def _cleanup(self):
        """Ensure all cache is flushed when the program exits."""
        if hasattr(self, '_trace_cache') and self._trace_cache:
            self.logger.info("ModelTracer cleanup running, flushing all requests.")
            self._flush_finished_requests()
    
    def __del__(self):
        """Attempt to flush on deletion, but don't rely on this being called."""
        try:
            self.logger.info("ModelTracer is being destroyed, attempting to flush requests.")
            self._cleanup()
        except Exception as e:
            # During shutdown, some modules might already be unloaded
            # so we can't rely on logging working properly
            pass


class ModelRunnerSim(ModelRunner):
    """
    A simulated model runner that uses a trace file to retrieve outputs instead of invoking the actual model.
    
    This class is used for testing a customized device backend and decoupling its functionality from performance. 
    """

    def __init__(self, *args, **kwargs):
        logger.warning(f"Simulating execution with {kwargs["server_args"].sim_gpu_memory}GB memory/worker.")
        self.sim_gpu_memory = kwargs["server_args"].sim_gpu_memory      # We could assume an GPU memory with arbitrary sizes

        super(ModelRunnerSim, self).__init__(*args, **kwargs)
        self._load_trace(ForwardResultTracer.trace_file_path)

    def _load_trace(self, trace_path) -> Dict[str, Dict]:
        if not os.path.exists(ForwardResultTracer.trace_file_path):
            raise RuntimeError(
                f"Trace file {ForwardResultTracer.trace_file_path} does not exist. "
            )
        self.traces = defaultdict(dict)
        with open(trace_path, "r") as f:
            for line in f:
                try:
                    for rid, next_token_ids in json.loads(line).items():
                        self.traces[rid] = next_token_ids
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON: {e}")
        logger.info(f"Loaded {len(self.traces)} traces from {trace_path}")

    # FIXME: I'm not sure if we have to override init_torch_distributed, which set the collectives at parent class initialization. 
    def initialize(self, min_per_gpu_memory: float):
        server_args = self.server_args
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        # Load the model
        self.sampler = Sampler()
        self.load_model()       # TODO: well, just in case of some strange invoking from the outside world like TPWorker.

        # Init memory pool and attention backends
        self.init_memory_pool(
            min_per_gpu_memory,
            server_args.max_running_requests,
            server_args.max_total_tokens,
        )

        # The device is "cuda" in simulation to enable model_loading without more changes, but we do not exactly invoke it. 
        self.cuda_graph_runner = None
        self.attn_backend = SimAttnBackend(self)


    def init_memory_pool(
        self,
        total_gpu_memory: int,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):

        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if is_hip():  # Using natively supported format
                self.kv_cache_dtype = torch.float8_e5m2fnuz
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            if is_cuda():
                self.kv_cache_dtype = torch.float8_e4m3fn
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

        self.max_total_num_tokens = self.profile_max_num_token(self.sim_gpu_memory)

        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(
                        self.max_total_num_tokens / self.model_config.context_len * 512
                    ),
                    2048,
                ),
                4096,
            )

        if SGLANG_CI_SMALL_KV_SIZE:
            self.max_total_num_tokens = int(SGLANG_CI_SMALL_KV_SIZE)

        if not self.spec_algorithm.is_none():
            raise RuntimeError("Spec algorithm is not supported in simulation mode.")            

        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                logging.warning(
                    f"max_total_tokens={max_total_tokens} is larger than the profiled value "
                    f"{self.max_total_num_tokens}. "
                    f"Use the profiled value instead."
                )
            self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)

        self.max_total_num_tokens = (
            self.max_total_num_tokens
            // self.server_args.page_size
            * self.server_args.page_size
        )

        if self.max_total_num_tokens <= 0:
            raise RuntimeError(
                "Not enough memory. Please try to increase --mem-fraction-static."
            )

        # We need to keep the tier-0 index for outsider invoking. 
        if self.req_to_token_pool is None:
            self.req_to_token_pool = ReqToTokenPool(
                size=max_num_reqs + 1,
                max_context_len=self.model_config.context_len + 4,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
            )
        else:
            # Draft worker shares req_to_token_pool with the target worker.
            assert self.is_draft_worker

        # FIXME: only support MHA now
        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and not self.server_args.disable_mla
        ):
            assert False
        
        if self.server_args.enable_double_sparsity:
            assert False
        
        if self.is_multimodal:
            assert False
    

        self.token_to_kv_pool = MHATokenToKVPoolSim(
            self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=self.model_config.head_dim,
            layer_num=self.model_config.num_hidden_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
        )

        if self.token_to_kv_pool_allocator is None:
            if self.page_size == 1:
                self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                    self.max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                )
            else:
                self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                )

    cnt = 0

    def forward(
        self, forward_batch: ForwardBatch, skip_attn_backend_init: bool = False
    ) -> LogitsProcessorOutput:
        # Get the info of forward_batch here
        # out_cache_loc_cpu = forward_batch.out_cache_loc.detach().cpu().tolist()
        kvc_indices = forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices]
        if self.cnt == 0:
            pickle.dump(kvc_indices, open("nonzero_counts.pkl", "wb"))
            self.cnt += 1
        elif self.cnt < 100:
            pickle.dump(kvc_indices, open("nonzero_counts.pkl", "ab"))
            self.cnt += 1
        
        if forward_batch.forward_mode.is_decode():
            pass
        elif forward_batch.forward_mode.is_extend():
            pass
        elif forward_batch.forward_mode.is_idle():
            return self.forward_idle(forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")
        
        return None
        # raise NotImplementedError()

    # Function simulation
    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
    
        next_token_ids = torch.zeros(len(forward_batch.req_pos), dtype=torch.int32, device=self.device)
        
        for i, (rid, output_loc) in enumerate(forward_batch.req_pos):
            next_token_ids[i] = self.traces[rid][output_loc]        # FIXME: Actually, the ``next-toekn`` refer to trace[output_loc + 1]. But that will result in an indexing error
                                                                    # for those requests that are truncated by the context limit, since the truncatation would happen after 
                                                                    # sampling. We choose to use trace[output_loc] instead trace[output_loc + 1] to avoid this issue. 

        return next_token_ids
        

    def profile_max_num_token(self, total_gpu_memory: int):
        available_gpu_memory = self.sim_gpu_memory - 10   # TODO: we need a virtual device to manage the simulated memory
        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and not self.server_args.disable_mla
        ):
            cell_size = (
                (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                * self.model_config.num_hidden_layers
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        else:
            cell_size = (
                self.model_config.get_num_kv_heads(get_attention_tp_size())
                * self.model_config.head_dim
                * self.model_config.num_hidden_layers
                * 2
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - self.mem_fraction_static
        )
        max_num_token = int(rest_memory * (1 << 30) // cell_size)
        return max_num_token
