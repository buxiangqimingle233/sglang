import requests
import subprocess
from sglang.utils import wait_for_server, print_highlight, terminate_process


port = 30000
# python3 bench_sglang.py --parallel 256 --port 30000 --data-path /mlx/users/wz.21/playground/huggingface/hub/datasets--opencompass--AIME2025 --question-key question --answer-key answer --num-tries 64
benchmark_script_path = "/mlx/users/wz.21/playground/sglang/benchmark/reasoning_benchmark/bench_sglang.py"
benchmark_data_path = "/mlx/users/wz.21/playground/huggingface/hub/datasets--opencompass--AIME2025/aime2025-II.jsonl"

log_path = "/home/wangzhao/sglang/server-debug.log"
model_path = "/datasets/zhao/Meta-Llama-3.1-8B-Instruct"

command = f"nohup python3 {benchmark_script_path} --parallel 256 --port {port} --data-path {benchmark_data_path} --question-key question --answer-key answer --num-tries 64 > sglang-client.log 2>&1 &"

process = subprocess.Popen(command, shell=True)
