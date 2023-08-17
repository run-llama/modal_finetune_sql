from typing import Optional

from modal import gpu, method, Retries
from modal.cls import ClsMixin
import json

from .common import (
    MODEL_PATH,
    output_vol,
    stub,
    VOL_MOUNT_PATH,
    get_model_path,
    generate_prompt_sql
)
from .inference_utils import OpenLlamaLLM

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
def run_query(query: str, model_dir: str = "data_sql", use_finetuned_model: bool = True):
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
        model_dir=model_dir, max_new_tokens=num_output, use_finetuned_model=use_finetuned_model
    )
    service_context = ServiceContext.from_defaults(llm=llm)

    sql_path = VOL_MOUNT_PATH / "test_data.db"
    engine = create_engine(f'sqlite:///{sql_path}', echo=True)
    sql_database = SQLDatabase(engine)

    # define custom text-to-SQL prompt with generate prompt
    prompt_prefix = "Dialect: {dialect}\n\n"
    prompt_suffix = generate_prompt_sql("{query_str}", "{schema}", output="")
    sql_prompt = Prompt(prompt_prefix + prompt_suffix)

    query_engine = NLSQLTableQueryEngine(
        sql_database, 
        text_to_sql_prompt=sql_prompt,
        service_context=service_context,
        synthesize_response=False
    )
    response = query_engine.query(query)

    return response


def print_response(response):
    print(
        f'*****Model output*****\n'
        f'SQL Query: {str(response.metadata["sql_query"])}\n'
        f"Response: {response.response}\n"
    )

@stub.local_entrypoint()
def main(query: str, sqlite_file_path: str, model_dir: str = "data_sql", use_finetuned_model: str = "True"):
    """Main function."""

    fp = open(sqlite_file_path, "rb")
    stub.data_dict["sqlite_data"] = fp.read()

    if use_finetuned_model == "None":
        # try both
        response_0 = run_query.call(query, model_dir=model_dir, use_finetuned_model=True)
        print_response(response_0)
        response_1 = run_query.call(query, model_dir=model_dir, use_finetuned_model=False)
        print_response(response_1)
    else:
        bool_toggle = use_finetuned_model == "True"
        response = run_query.call(query, model_dir=model_dir, use_finetuned_model=bool_toggle)
        print_response(response)
