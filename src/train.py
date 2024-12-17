import litgpt
from lightning import Trainer
from litgpt.lora import merge_lora_weights

from model.lora import FinetuneLLM
from data.loader import HFDataLoader

model = "meta-llama/Llama-3.2-1B-Instruct"
path = "hugfaceguy0001/retarded_bar"
litgpt.LLM.load(model=model)

tokenizer = litgpt.Tokenizer(f"checkpoints/{model}")

dataset = HFDataLoader(path=path, name="question", split="train", question="text", answer="answer")
data = dataset.load_as_json()
data.connect(tokenizer, batch_size=1, max_seq_length=512)

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=720,
    accumulate_grad_batches=8,
    precision="bf16-true",
    log_every_n_steps=10,
)
with trainer.init_module(empty_init=True):
    finetuned_llm = FinetuneLLM(model=model)

trainer.fit(finetuned_llm, data)

# Save final checkpoint
merge_lora_weights(finetuned_llm.model)
trainer.save_checkpoint("checkpoints/finetuned.ckpt", weights_only=True)
