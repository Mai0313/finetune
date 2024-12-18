import hydra
import torch
import litgpt
from lightning import Trainer
from omegaconf import OmegaConf, DictConfig
from litgpt.lora import merge_lora_weights
from rich.console import Console

from model.lora import FinetuneLLM
from data.loader import HFDataLoader
from utils.instantiators import instantiate_loggers, instantiate_callbacks

console = Console()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def train(config: DictConfig) -> None:
    console.print(OmegaConf.to_container(config))

    tokenizer = litgpt.Tokenizer(f"checkpoints/{config.pretrained.model}")

    dataset = HFDataLoader(**config.data)
    loaded_data = dataset.load_as_json()
    loaded_data.connect(
        tokenizer=tokenizer,
        batch_size=config.data.batch_size,
        max_seq_length=config.data.max_seq_length,
    )

    callbacks = instantiate_callbacks(config.callbacks)

    logger = instantiate_loggers(config.logger)

    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(**config.trainer, callbacks=callbacks, logger=logger)

    with trainer.init_module(empty_init=True):
        model = FinetuneLLM(**config.pretrained)

    trainer.fit(model, loaded_data)

    # Save final checkpoint
    merge_lora_weights(model.model)
    model_name = config.pretrained.model.split("/")[-1]
    ckpt_name = f"{model_name}-finetuned.ckpt"
    trainer.save_checkpoint(f"checkpoints/{ckpt_name}", weights_only=True)


if __name__ == "__main__":
    train()
