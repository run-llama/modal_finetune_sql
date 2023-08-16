from typing import Optional

from modal import gpu, method, Retries
from modal.cls import ClsMixin
import json

from .common import (
    MODEL_PATH,
    output_vol,
    stub,
    VOL_MOUNT_PATH,
    user_model_path,
    generate_prompt_sql
)

from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    CustomLLM, 
    LLMMetadata, 
    CompletionResponse, 
    CompletionResponseGen,
)
from llama_index.llms.base import llm_completion_callback
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index import SQLDatabase, ServiceContext, Prompt
from typing import Any


@stub.cls(
    gpu=gpu.A100(memory=20),
    network_file_systems={VOL_MOUNT_PATH: output_vol},
)
class OpenLlamaLLM(CustomLLM, ClsMixin):
    """OpenLlamaLLM is a custom LLM that uses the OpenLlamaModel."""

   
    def __init__(
        self, 
        user: str, 
        max_new_tokens: int = 128,
        callback_manager: Optional[CallbackManager] = None,
        use_finetuned_model: bool = True,
    ):
        super().__init__(callback_manager=callback_manager)

        import sys

        import torch
        from peft import PeftModel
        from transformers import LlamaForCausalLM, LlamaTokenizer

        self.user = user
        CHECKPOINT = user_model_path(self.user)

        load_8bit = False
        device = "cuda"

        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)

        model = LlamaForCausalLM.from_pretrained(
            MODEL_PATH,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if use_finetuned_model:
            model = PeftModel.from_pretrained(
                model,
                CHECKPOINT,
                torch_dtype=torch.float16,
            )

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        self.model = model
        self.device = device

        self._max_new_tokens = max_new_tokens

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=2048,
            num_output=self._max_new_tokens,
            model_name="finetuned_openllama_sql"
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        import torch
        from transformers import GenerationConfig
        # TODO: TO fill
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # print(tokens)
        generation_config = GenerationConfig(
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=self._max_new_tokens,
            )

        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s, skip_special_tokens=True)
        # NOTE: parsing response this way means that the model can mostly
        # only be used for text-to-SQL, not other purposes
        response_text = output.split("### Response:")[1].strip()
        return CompletionResponse(text=response_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()


@stub.function(
    gpu="A100",
    retries=Retries(
        max_retries=3,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
    timeout=60 * 60 * 2,
    network_file_systems={VOL_MOUNT_PATH.as_posix(): output_vol},
    cloud="gcp",
)
def run_query(user: str, query: str, use_finetuned_model: bool = True):
    """Run query."""
    import pandas as pd
    from sqlalchemy import create_engine

    # define SQL database
    assert "sqlite_data" in stub.data_dict
    with open(VOL_MOUNT_PATH / "test_data.db", "wb") as fp:
        fp.write(stub.data_dict["sqlite_data"])
    
    # define service context (containing custom LLM)
    print('setting up service context')
    # finetuned llama LLM
    num_output = 256
    llm = OpenLlamaLLM(
        user=user, max_new_tokens=num_output, use_finetuned_model=use_finetuned_model
    )
    # # default Llama LLM (load from our huggingface loader)
    # default_llm = HuggingFaceLLM(
    #     context_window=2048,
    #     max_new_tokens=num_output,
    #     tokenizer_name="openlm-research/open_llama_7b",
    #     model_name="openlm-research/open_llama_7b"
    # )
    service_context = ServiceContext.from_defaults(llm=llm)
    # default_service_context = ServiceContext.from_defaults(llm=default_llm)

    sql_path = VOL_MOUNT_PATH / "test_data.db"
    engine = create_engine(f'sqlite:///{sql_path}', echo=True)
    sql_database = SQLDatabase(engine)

    # define custom text-to-SQL prompt with generate prompt
    prompt_prefix = "Dialect: {dialect}\n\n"
    prompt_suffix = generate_prompt_sql(user, "{query_str}", "{schema}", output="")
    sql_prompt = Prompt(prompt_prefix + prompt_suffix)

    query_engine = NLSQLTableQueryEngine(
        sql_database, 
        text_to_sql_prompt=sql_prompt,
        service_context=service_context,
        synthesize_response=False
    )
    response = query_engine.query(query)

    # # give baseline response too
    # default_query_engine = NLSQLTableQueryEngine(
    #     sql_database, 
    #     text_to_sql_prompt=sql_prompt,
    #     service_context=default_service_context,
    #     synthesize_response=False
    # )
    # default_response = default_query_engine.query(query)

    print(f'Model output: {str(response.metadata["sql_query"])}')

    return response

@stub.local_entrypoint()
def main(user: str, query: str, sqlite_file_path: str, use_finetuned_model: Optional[bool] = None):
    """Main function."""

    fp = open(sqlite_file_path, "rb")
    stub.data_dict["sqlite_data"] = fp.read()

    if use_finetuned_model is None:
        # try both
        run_query.call(user, query, use_finetuned_model=True)
        run_query.call(user, query, use_finetuned_model=False)
    else:
        run_query.call(user, query, use_finetuned_model=use_finetuned_model)
