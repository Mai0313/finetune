import torch
from lightning import LightningModule
from litgpt.lora import GPT, mark_only_lora_as_trainable
from litgpt.utils import chunked_cross_entropy


class FinetuneLLM(LightningModule):
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
        mark_only_lora_as_trainable(self.model)

    def on_train_start(self) -> None:
        ckpt_path = f"checkpoints/{self.model_name}/lit_model.pth"
        state_dict = torch.load(ckpt_path, mmap=True, weights_only=False)
        self.model.load_state_dict(state_dict, strict=False)

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
