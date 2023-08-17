"""Get inference utils."""

from typing import Optional

from modal import gpu
from modal.cls import ClsMixin

from .common import (
    MODEL_PATH,
    output_vol,
    stub,
    VOL_MOUNT_PATH,
    get_model_path,
)

from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    CustomLLM, 
    LLMMetadata, 
    CompletionResponse, 
    CompletionResponseGen,
)
from llama_index.llms.base import llm_completion_callback
from typing import Any

@stub.cls(
    gpu=gpu.A100(memory=20),
    network_file_systems={VOL_MOUNT_PATH: output_vol},
)
class OpenLlamaLLM(CustomLLM, ClsMixin):
    """OpenLlamaLLM is a custom LLM that uses the OpenLlamaModel."""

   
    def __init__(
        self, 
        model_dir: str = "data_sql",
        max_new_tokens: int = 128,
        callback_manager: Optional[CallbackManager] = None,
        use_finetuned_model: bool = True,
    ):
        super().__init__(callback_manager=callback_manager)

        import sys

        import torch
        from peft import PeftModel
        from transformers import LlamaForCausalLM, LlamaTokenizer

        CHECKPOINT = get_model_path(model_dir)

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

