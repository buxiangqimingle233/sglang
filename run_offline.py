# launch the offline engine
import os

# from sglang.srt.conversation import chat_templates
# from sglang.test.test_utils import is_in_ci
# from sglang.utils import async_stream_and_merge, stream_and_merge

# hf_root = os.environ.get("HF_ROOT", "/mlx_devbox/users/wz.21/playground/huggingface/hub")
hf_root = os.environ.get("HF_ROOT", "/datasets/zhao")
cot_llama_path = os.path.join(hf_root, "models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B")


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is", 
]
rids = [str(i) for i in range(len(prompts))]

def setup_gpu_env():
    # set system path
    os.environ["PATH"] = "/usr/local/cuda-12.1/bin:" + os.environ["PATH"]

def setup_cpu_env():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  

def trace_execution():
    setup_gpu_env()
    
    import sglang as sgl
    
    config = {
        "model_path": cot_llama_path,
        "tp_size": 4,
        "base_gpu_id": 0,
        "disable_overlap_schedule": True,
        "log_level": "info",
        "enable_forward_result_tracing": True,
        "enable_cuda_graph_dump": False,
        "disable_cuda_graph": True,     # FIXME: The flashinfer backend has something wrong with cuda graph (@cuda121, torch251)
        "trace_file": "./trace/test_trace.jsonl"
    }
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 1024 * 8}    # CoT models love speaking
    
    # Tracing the forward result
    llm = sgl.Engine(**config)

    outputs = llm.generate(prompts, sampling_params, rid=rids)

    for rid, output in zip(rids, outputs):
        print("===============================")
        print(f"RID: {rid} input-length: f{len(prompts[int(rid)])} output-length: {len(output['text'])}")


def replay():
    setup_cpu_env()
    
    import sglang as sgl
    
    config = {
        "model_path": cot_llama_path,
        "tp_size": 8,
        "device": "cpu",
        "attention_backend": "torch_native",
        "base_gpu_id": 0,
        "disable_overlap_schedule": True,
        "log_level": "info",
        "disable_cuda_graph": True,
        "enable_model_runner_sim": True,
        "enable_forward_result_tracing": False,
        "enable_cuda_graph_dump": False,
        "sim_gpu_memory": 1024,
        "trace_file": "./trace/am-sample.jsonl"
    }
    
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 1024 * 8}    # Well, CoT models like speaking too much

    llm = sgl.Engine(**config)
    outputs = llm.generate(prompts, sampling_params, rid=rids)
    

def replay_llama_distill():
    import sglang as sgl
    
    config = {
        "model_path": cot_llama_path,
        "tp_size": 1,
        "base_gpu_id": 2,
        "disable_overlap_schedule": True,
        "log_level": "info",
        "disable_cuda_graph": True,
        "enable_model_runner_sim": True,
        "enable_forward_result_tracing": False,
        "enable_cuda_graph_dump": False,
        "sim_gpu_memory": 1024,
        "trace_file": "./trace/am-sample.jsonl"
    }
    
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 1024 * 8}    # Well, CoT models like speaking too much

    llm = sgl.Engine(**config)
    outputs = llm.generate(prompts, sampling_params, rid=rids)


def main():
    # trace_execution()
    replay()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
