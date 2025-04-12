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
import torch
import time

from typing import List, Optional, Tuple, Union, Dict, Any
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


class ForwardResultTracer:
    """
    A tracer module for capturing model outputs.
    
    This class is responsible for tracking outputs from model execution and persisting
    them to disk. It operates independently from the model runner to avoid modifying
    the original execution flow.
    """
    
    def __init__(self, enable_execution_tracing: bool = False, tp_rank: int = 0):
        self.should_trace = enable_execution_tracing and tp_rank == 0  # Only log on rank 0 to avoid duplication
        self.trace_dir = "trace"
        self.tp_rank = tp_rank
        self.trace_file = None  # Will be initialized at first trace call

        self._trace_cache = defaultdict(dict)  # Using request ID -> position -> output_ids
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

            # Add next_token_ids to the cache, indexed by request ID and position
            for req, next_token_id in zip(schedule_batch.reqs, next_token_ids):
                self._trace_cache[req.rid][len(req.output_ids)] = next_token_id
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
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir, exist_ok=True)
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        timestamp = ""
        self.trace_file = f"{self.trace_dir}/model_trace_{timestamp}_tp{self.tp_rank}.jsonl"
        open(self.trace_file, "w").close()                      # Create an empty file
        self.logger.info(f"Trace stored to {self.trace_file}")
    
    def _mark_request_completed(self, request_id: str) -> None:
        """Mark a request as completed so it can be flushed to disk."""
        if request_id in self._trace_cache and request_id not in self._finished_requests:
            self._finished_requests.add(request_id)
            self.logger.info(f"Request {request_id} marked as completed")

    def _flush_finished_requests(self) -> None:
        if not self._finished_requests or not self.trace_file:
            return
        
        requests_to_flush = self._finished_requests.intersection(self._trace_cache.keys())
        if not requests_to_flush:
            return
        self.logger.info(f"Flushing {len(requests_to_flush)} completed requests to {self.trace_file}")

        try:
            with open(self.trace_file, 'a') as f:
                for rid in requests_to_flush:
                    f.write(json.dumps({rid: self._trace_cache[rid]}) + "\n")
                    # Remove from cache after flushing
                    self._trace_cache.pop(rid)
                    self._finished_requests.remove(rid)
        except Exception as e:
            self.logger.error(f"Error writing trace file: {e}")

    def cleanup(self):
        """Ensure all cache is flushed when the program exits."""
        if hasattr(self, '_trace_cache') and self._trace_cache:
            self.logger.info("ModelTracer cleanup running, flushing all requests.")
            # Initialize trace file if it hasn't been done yet (in case no tracing occurred)
            if not self.trace_file and self.should_trace:
                self._initialize_trace_file()
            self._flush_all_requests()
    
    def __del__(self):
        """Attempt to flush on deletion, but don't rely on this being called."""
        try:
            self.logger.info("ModelTracer is being destroyed, attempting to flush requests.")
            self._cleanup()
        except Exception as e:
            # During shutdown, some modules might already be unloaded
            # so we can't rely on logging working properly
            pass