from pydantic import BaseModel


class Trainer(BaseModel):
    accelerator: str
    strategy: str
    devices: str
    check_val_every_n_epoch: int
    deterministic: bool
    max_epochs: int
    min_epochs: int
    enable_progress_bar: bool
    default_root_dir: str


class Pretrained(BaseModel):
    model: str


class Dataset(BaseModel):
    data: str
    name: str
    split: str
    question: str
    answer: str


class FinetuneConfig(BaseModel):
    task_name: str
    loggers: list[str]
    tags: list[str]
    trainer: Trainer
    pretrained: Pretrained
    dataset: Dataset
