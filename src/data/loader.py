from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from pydantic import Field, BaseModel
from litgpt.data import JSON


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
            cache_dir="./data/tmp",
            num_proc=self.max_cpu,
        )
        dataset = dataset.rename_columns({self.question: "instruction", self.answer: "output"})
        need_to_remove = [
            col for col in dataset.column_names if col not in ["instruction", "output"]
        ]
        dataset = dataset.remove_columns(need_to_remove)
        return dataset

    def load_list(self) -> list[dict[str, Any]]:
        dataset = self.load()
        return dataset.to_list()

    def load_as_json(self, val_split: float = 0.2) -> JSON:
        dataset = self.load()
        filepath = Path(dataset.cache_files[0]["filename"])
        dataset_path = Path(f"{filepath.parent}/{filepath.stem}.json")
        dataset.to_json(
            dataset_path.as_posix(), num_proc=self.max_cpu, force_ascii=False, lines=False
        )
        return JSON(json_path=dataset_path, val_split_fraction=val_split)
