from typing import Optional
from pathlib import Path

from .common import VOL_MOUNT_PATH

def user_data_path(user: str, data_dir: str = "data_sql") -> Path:
    return VOL_MOUNT_PATH / data_dir / user / "data_sql.jsonl"

def user_model_path(user: str, data_dir: str = "data_sql", checkpoint: Optional[str] = None) -> Path:
    path = VOL_MOUNT_PATH / data_dir / user
    if checkpoint:
        path = path / checkpoint
    return path

def generate_prompt_sql(user, input, context, output=""):
    return f"""You are {user}, a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. 

You must output the SQL query that answers the question.

### Input:
{input}

### Context:
{context}

### Response:
{output}"""