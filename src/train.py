import litgpt
from lightning import Trainer
from litgpt.lora import merge_lora_weights

from model.lora import FinetuneLLM
from data.loader import HFDataLoader


def train(model: str, data: str, name: str, split: str, question: str, answer: str) -> None:
    litgpt.LLM.load(model=model)

    tokenizer = litgpt.Tokenizer(f"checkpoints/{model}")

    dataset = HFDataLoader(path=data, name=name, split=split, question=question, answer=answer)
    loaded_data = dataset.load_as_json()
    loaded_data.connect(tokenizer, batch_size=1, max_seq_length=512)

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

    trainer.fit(finetuned_llm, loaded_data)

    # Save final checkpoint
    merge_lora_weights(finetuned_llm.model)
    model_name = model.split("/")[-1]
    ckpt_name = f"{model_name}-finetuned.ckpt"
    trainer.save_checkpoint(f"checkpoints/{ckpt_name}", weights_only=True)


if __name__ == "__main__":
    model = "meta-llama/Llama-3.2-1B-Instruct"
    data = "hugfaceguy0001/retarded_bar"
    name = "question"
    split = "train"
    question = "text"
    answer = "answer"
    train(model=model, data=data, name=name, split=split, question=question, answer=answer)
