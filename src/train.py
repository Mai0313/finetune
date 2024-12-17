import json

import hydra
import litgpt
from lightning import Trainer
from omegaconf import OmegaConf, DictConfig
from litgpt.lora import merge_lora_weights

from model.lora import FinetuneLLM
from data.loader import HFDataLoader
from _types.config import FinetuneConfig


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def train(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(cfg)
    with open("config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    config = FinetuneConfig(**config_dict)
    litgpt.LLM.load(model=config.pretrained.model)

    tokenizer = litgpt.Tokenizer(f"checkpoints/{config.pretrained.model}")

    dataset = HFDataLoader(
        path=config.dataset.data,
        name=config.dataset.name,
        split=config.dataset.split,
        question=config.dataset.question,
        answer=config.dataset.answer,
    )
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
        finetuned_llm = FinetuneLLM(model=config.pretrained.model)

    trainer.fit(finetuned_llm, loaded_data)

    # Save final checkpoint
    merge_lora_weights(finetuned_llm.model)
    model_name = config.pretrained.model.split("/")[-1]
    ckpt_name = f"{model_name}-finetuned.ckpt"
    trainer.save_checkpoint(f"checkpoints/{ckpt_name}", weights_only=True)


if __name__ == "__main__":
    train()
