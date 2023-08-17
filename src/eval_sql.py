from modal import Retries

from .common import (
    output_vol,
    stub,
    VOL_MOUNT_PATH,
    get_data_path,
    generate_prompt_sql
)
from .inference_utils import OpenLlamaLLM


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
def run_evals(
    sample_data, 
    model_dir: str = "data_sql", 
    use_finetuned_model: bool = True
):
    llm = OpenLlamaLLM(
        model_dir=model_dir, max_new_tokens=256, use_finetuned_model=use_finetuned_model
    )
    inputs_outputs = []
    for row_dict in sample_data:
        prompt = generate_prompt_sql(row_dict["input"], row_dict["context"])
        completion = llm.complete(
            prompt,
            do_sample=True,
            temperature=0.3,
            top_p=0.85,
            top_k=40,
            num_beams=1,
            max_new_tokens=600,
            repetition_penalty=1.2,
        )
        inputs_outputs.append((row_dict, completion.text))
    return inputs_outputs


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
def run_evals_all(
    data_dir: str = "data_sql", 
    model_dir: str = "data_sql", 
    num_samples: int = 10, 
):
    # evaluate a sample from the same training set
    from datasets import load_dataset

    data_path = get_data_path(data_dir).as_posix()
    data = load_dataset("json", data_files=data_path)

    # load sample data
    sample_data = data["train"].shuffle().select(range(num_samples))

    print('*** Running inference with finetuned model ***')
    inputs_outputs_0 = run_evals(
        sample_data=sample_data, 
        model_dir=model_dir, 
        use_finetuned_model=True
    )

    print('*** Running inference with base model ***')
    input_outputs_1 = run_evals(
        sample_data=sample_data, 
        model_dir=model_dir, 
        use_finetuned_model=False
    )

    return inputs_outputs_0, input_outputs_1



@stub.local_entrypoint()
def main(data_dir: str = "data_sql", model_dir: str = "data_sql", num_samples: int = 10):
    """Main function."""
    inputs_outputs_0, input_outputs_1 = run_evals_all.call(
        data_dir=data_dir,
        model_dir=model_dir,
        num_samples=num_samples
    )
    for idx, (row_dict, completion) in enumerate(inputs_outputs_0):
        print(f'************ Row {idx} ************')
        print(f"Input {idx}: " + str(row_dict))
        print(f"Output {idx} (finetuned model): " + str(completion))
        print(f"Output {idx} (base model): " + str(input_outputs_1[idx][1]))
        print('***********************************')
