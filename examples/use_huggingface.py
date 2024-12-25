import json

import torch
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

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,
)

result = pipe("./data/tmp/sample_41.mp3")
chunks = result["chunks"]

with open("result.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False)
