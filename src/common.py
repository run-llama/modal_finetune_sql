from modal import Image, Stub, NetworkFileSystem, Dict
import random
from typing import Optional
from pathlib import Path

VOL_MOUNT_PATH = Path("/vol")

WANDB_PROJECT = "test-finetune-modal"

MODEL_PATH = "/model"


def download_models():
    from transformers import LlamaForCausalLM, LlamaTokenizer

    model_name = "openlm-research/open_llama_7b"

    model = LlamaForCausalLM.from_pretrained(model_name)
    model.save_pretrained(MODEL_PATH)

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(MODEL_PATH)


openllama_image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.7",
        "cudnn=8.1.0",
        "cuda-nvcc",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "accelerate==0.18.0",
        "bitsandbytes==0.37.0",
        "bitsandbytes-cuda117==0.26.0.post2",
        "datasets==2.10.1",
        "fire==0.5.0",
        "gradio==3.23.0",
        "peft @ git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08",
        "transformers @ git+https://github.com/huggingface/transformers.git@a92e0ad2e20ef4ce28410b5e05c5d63a5a304e65",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "sentencepiece==0.1.97",
        "llama-index==0.8.1",
        "sentence-transformers",
    )
    .run_function(download_models)
    .pip_install("wandb==0.15.0")
)

stub = Stub(name="doppel-bot", image=openllama_image)
stub.model_dict = Dict.new()
stub.data_dict = Dict.new()

output_vol = NetworkFileSystem.new(cloud="gcp").persisted("doppelbot-vol")


def generate_prompt_sql(user, input, context, output=""):
    return f"""You are {user}, a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. 

You must output the SQL query that answers the question.

### Input:
{input}

### Context:
{context}

### Response:
{output}"""


def user_data_path(user: str, data_dir: str = "data_sql") -> Path:
    return VOL_MOUNT_PATH / data_dir / user / "data_sql.jsonl"

def user_model_path(user: str, data_dir: str = "data_sql", checkpoint: Optional[str] = None) -> Path:
    path = VOL_MOUNT_PATH / data_dir / user
    if checkpoint:
        path = path / checkpoint
    return path

def get_user_for_team_id(team_id: Optional[str], users: list[str]) -> Optional[str]:
    # Dumb: for now, we only allow one user per team.
    path = VOL_MOUNT_PATH / (team_id or "data")
    filtered = []
    for p in path.iterdir():
        # Check if finished fine-tuning.
        if (path / p / "adapter_config.json").exists() and p.name in users:
            filtered.append(p.name)
    if not filtered:
        return None
    user = random.choice(filtered)
    print(f"Randomly picked {user} out of {filtered}.")
    return user
