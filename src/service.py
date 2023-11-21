from __future__ import annotations

import bentoml
import openllm
from bentoml.io import JSON, NumpyNdarray
from transformers import AutoTokenizer, AutoModel
from typing import List, TYPE_CHECKING

from .embedding_runnable import SentenceEmbeddingRunnable

if TYPE_CHECKING:
    import numpy.typing as npt

verbose = False

llm_model = "TheBloke/Llama-2-7B-fp16"
llm_adapter = "IlyaGusev/saiga2_7b_lora"

# Embedding model
embedding_model="cointegrated/rubert-tiny2"


tokenizer = AutoTokenizer.from_pretrained(embedding_model)
model = AutoModel.from_pretrained(embedding_model)

bentoml.transformers.save_model("rubert-tiny2", model)
bentoml.transformers.save_model("rubert-tiny2-tokenizer", tokenizer)

llm = openllm.LLM(llm_model, backend='pt', adapter_map={llm_adapter:"ru"})
llm.save_pretrained()

embed_runner = bentoml.Runner(
    SentenceEmbeddingRunnable,
    name='sentence_embedding_model',
    max_batch_size=32,
    max_latency_ms=300
)

svc = bentoml.Service(name="rulawbot-service", runners=[llm.runner, embed_runner])

@svc.on_startup
def download(_: bentoml.Context):
    llm.runner.download_model()


@svc.api(
        input=bentoml.io.Text(), 
        output=bentoml.io.Text())
async def prompt(input_text: str) -> str:
    if verbose:
        print(f"GOT TEXT: {input_text}", flush=True, end='')
    answer = await llm.generate(input_text, adapter_name="ru", stop_token_ids=[2], max_new_tokens=256)
    if verbose:
        print(f"GEN TEXT: {answer.outputs[0].text}", flush=True, end='')
    return answer.outputs[0].text

@svc.api(
    input=bentoml.io.Text(),
    output=NumpyNdarray()
)
async def encode(docs: str):
    return await embed_runner.encode.async_run([docs]) 
