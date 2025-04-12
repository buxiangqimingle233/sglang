# launch the offline engine
import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge

def main():
    # For debugging
    # model_path_ = "/datasets/zhao/MicroLlama"       # Shit model, just for debugging
    # llm = sgl.Engine(model_path=model_path_, tp_size=1, base_gpu_id=2, disable_overlap_schedule=True, log_level="info", enable_forward_result_tracing=True)

    # model_path_ = "/datasets/zhao/Meta-Llama-3.1-8B-Instruct"
    # llm = sgl.Engine(model_path=model_path_, tp_size=4, base_gpu_id=0, disable_overlap_schedule=True, log_level="info", enable_forward_result_tracing=True)

    model_path_ = "/datasets/zhao/DeepSeek-R1-Distill-Llama-8B"
    llm = sgl.Engine(model_path=model_path_, tp_size=2, base_gpu_id=2, disable_overlap_schedule=True, log_level="info", enable_forward_result_tracing=True)


    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 1024}

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
