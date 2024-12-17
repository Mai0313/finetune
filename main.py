import torch
import litgpt
from lightning import Trainer, LightningModule
from litgpt.lora import GPT, merge_lora_weights
import litgpt.model


class LitLLM(LightningModule):
    def __init__(self, model: str):
        super().__init__()
        self.model_name = model
        self.model = GPT.from_name(
            name=model,
            lora_r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_query=True,
            lora_key=False,
            lora_value=True,
        )
        litgpt.lora.mark_only_lora_as_trainable(self.model)

    def on_train_start(self) -> None:
        state_dict = torch.load(
            f"checkpoints/{self.model_name}/lit_model.pth", mmap=True, weights_only=False
        )
        self.model.load_state_dict(state_dict, strict=False)

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
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


if __name__ == "__main__":
    from pathlib import Path

    from datasets import load_dataset
    from litgpt.data import JSON

    data_path = "hugfaceguy0001/retarded_bar"
    dataset = load_dataset(
        path=data_path, name="question", split="train", cache_dir="./data", num_proc=8
    )
    filepath = Path(dataset.cache_files[0]["filename"])
    dataset.rename_columns({"text": "input", "answer": "output"})
    dataset.to_json(f"{filepath.parent}/{filepath.stem}.json", force_ascii=False)

    data = JSON(json_path=f"{filepath.parent}/{filepath.stem}.json")

    model = "meta-llama/Llama-3.2-3B-Instruct"
    litgpt.LLM.load(model=model)

    tokenizer = litgpt.Tokenizer(f"checkpoints/{model}")
    data.connect(tokenizer, batch_size=1, max_seq_length=512)

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=720,
        accumulate_grad_batches=8,
        precision="bf16-true",
    )
    with trainer.init_module(empty_init=True):
        model = LitLLM(model=model)

    trainer.fit(model, data)

    # Save final checkpoint
    merge_lora_weights(model.model)
    trainer.save_checkpoint("checkpoints/finetuned.ckpt", weights_only=True)
