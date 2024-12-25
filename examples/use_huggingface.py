import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    WhisperForConditionalGeneration,
    pipeline,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model: WhisperForConditionalGeneration = AutoModelForSpeechSeq2Seq.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     torch_dtype=torch_dtype,
#     device=device,
#     return_timestamps=True
# )

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

# result = pipe("./data/tmp/sample_41.mp3")
# print(result["text"])
