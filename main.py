from pathlib import Path

import torch
import litgpt
from datasets import Dataset, load_dataset
from pydantic import Field, BaseModel
from lightning import Trainer, LightningModule
from litgpt.data import JSON
from litgpt.lora import GPT, mark_only_lora_as_trainable
from litgpt.utils import chunked_cross_entropy


class LitLLM(LightningModule):
    def __init__(self, model: str):
        super().__init__()
        self.model = GPT.from_name(
            name=model,
            lora_r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_query=True,
            lora_key=False,
            lora_value=True,
        )
        self.ckpt = torch.load(f"checkpoints/{model}/lit_model.pth", mmap=True, weights_only=False)
        mark_only_lora_as_trainable(self.model)

    def on_train_start(self) -> None:
        self.model.load_state_dict(self.ckpt, strict=False)

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = self.model(input_ids)
        loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LambdaLR]]:
        warmup_steps = 10
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]


class HFDataLoader(BaseModel):
    # For Loading the dataset from huggingface
    path: str = Field(
        ...,
        title="The path to the dataset",
        description="The path/url to the dataset from huggingface.",
        frozen=True,
        deprecated=False,
    )
    name: str = Field(
        ...,
        title="The name of the dataset",
        description="The name of the dataset from huggingface.",
        frozen=True,
        deprecated=False,
    )
    split: str = Field(
        "train",
        title="The split of the dataset",
        description="The split of the dataset from huggingface.",
        frozen=True,
        deprecated=False,
    )

    # For Litgpt Rules; rename columns to required names
    question: str = Field(
        ...,
        title="The Question of the dataset",
        description="This is the column name of the question, this field will be renamed to `input`.",
        frozen=True,
        deprecated=False,
    )
    answer: str = Field(
        ...,
        title="The Answer of the dataset",
        description="This is the column name of the answer, this field will be renamed to `output`.",
        frozen=True,
        deprecated=False,
    )
    max_cpu: int = Field(
        default=8,
        title="The maximum number of CPU cores",
        description="The maximum number of CPU cores to use for loading the dataset.",
        frozen=False,
        deprecated=False,
    )

    def load(self) -> Dataset:
        dataset = load_dataset(
            path=self.path,
            name=self.name,
            split=self.split,
            cache_dir="./data",
            num_proc=self.max_cpu,
        )
        dataset = dataset.rename_columns({self.question: "instruction", self.answer: "output"})
        need_to_remove = [
            col for col in dataset.column_names if col not in ["instruction", "output"]
        ]
        dataset = dataset.remove_columns(need_to_remove)
        return dataset

    def load_as_json(self, val_split: float = 0.2) -> JSON:
        dataset = self.load()
        filepath = Path(dataset.cache_files[0]["filename"])
        dataset_path = Path(f"{filepath.parent}/{filepath.stem}.json")
        dataset.to_json(
            dataset_path.as_posix(), num_proc=self.max_cpu, force_ascii=False, lines=False
        )
        return JSON(json_path=dataset_path, val_split_fraction=val_split)


if __name__ == "__main__":
    from litgpt.lora import merge_lora_weights

    model = "meta-llama/Llama-3.2-1B-Instruct"
    path = "hugfaceguy0001/retarded_bar"
    litgpt.LLM.load(model=model)

    tokenizer = litgpt.Tokenizer(f"checkpoints/{model}")

    dataset = HFDataLoader(
        path=path, name="question", split="train", question="text", answer="answer"
    )
    data = dataset.load_as_json()
    data.connect(tokenizer, batch_size=1, max_seq_length=512)

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=720,
        accumulate_grad_batches=8,
        precision="bf16-true",
    )
    with trainer.init_module(empty_init=True):
        finetuned_llm = LitLLM(model=model)

    trainer.fit(finetuned_llm, data)

    # Save final checkpoint
    merge_lora_weights(finetuned_llm.model)
    trainer.save_checkpoint("checkpoints/finetuned.ckpt", weights_only=True)
