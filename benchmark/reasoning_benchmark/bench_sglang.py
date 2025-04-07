# launch the offline engine
import asyncio
import io
import os

from PIL import Image
import requests
import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge

if is_in_ci():
    import patch


llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")

