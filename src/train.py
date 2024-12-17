import torch
import litgpt
from lightning import Trainer, LightningModule
from litgpt.lora import GPT, mark_only_lora_as_trainable
from litgpt.utils import chunked_cross_entropy


class FinetuneLLM(LightningModule):
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


if __name__ == "__main__":
    from litgpt.lora import merge_lora_weights

    from data.loader import HFDataLoader

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
        log_every_n_steps=10,
    )
    with trainer.init_module(empty_init=True):
        finetuned_llm = FinetuneLLM(model=model)

    trainer.fit(finetuned_llm, data)

    # Save final checkpoint
    merge_lora_weights(finetuned_llm.model)
    trainer.save_checkpoint("checkpoints/finetuned.ckpt", weights_only=True)
