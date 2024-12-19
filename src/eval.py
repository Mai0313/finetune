from typing import Any, Literal

import hydra
import litgpt
from omegaconf import OmegaConf, DictConfig
from rich.console import Console
from rich.progress import Progress
from litgpt.prompts import Alpaca
from huggingface_hub import InferenceClient

from data.loader import HFDataLoader

console = Console()


def generate_model_scores(
    testing_datasets: list[dict[str, Any]],
    eval_model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    question_field: Literal["instruction"] = "instruction",
    target_field: Literal["output"] = "output",
    response_field: Literal["response"] = "response",
) -> list[int]:
    scores = []
    client = InferenceClient()
    with Progress() as progress:
        task = progress.add_task("Scoring responses", total=len(testing_datasets))
        for testing_dataset in testing_datasets:
            prompt = (
                f"Given the input `{testing_dataset[question_field]}` "
                f"and correct output `{testing_dataset[target_field]}`, "
                f"score the model response `{testing_dataset[response_field]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            completion = client.chat.completions.create(
                model=eval_model, messages=messages, max_tokens=500
            )
            score = completion.choices[0].message.content
            try:
                scores.append(int(score))
            except ValueError:
                continue
            progress.update(task, advance=1, description=f"Score: {score}")
    return scores


@hydra.main(version_base="1.3", config_path="../config", config_name="eval.yaml")
def start_eval(config: DictConfig) -> None:
    console.print(OmegaConf.to_container(config))
    prompt_style = Alpaca()
    llm = litgpt.LLM.load(config.pretrained.model)

    dataset = HFDataLoader(**config.data)
    testing_datasets = dataset.load_list()
    testing_datasets = [testing_datasets[0]]
    with Progress() as progress:
        task = progress.add_task("Generating responses", total=len(testing_datasets))
        for testing_dataset in testing_datasets:
            prompt = prompt_style.apply(prompt=testing_dataset["instruction"], **testing_dataset)
            response = llm.generate(prompt=prompt)
            testing_dataset["response"] = response
            progress.update(task, advance=1)
    del llm
    scores = generate_model_scores(testing_datasets=testing_datasets)
    console.print(f"Number of scores: {len(scores)} of {len(testing_datasets)}")
    console.print(f"Average score: {sum(scores) / len(scores):.2f}\n")


if __name__ == "__main__":
    start_eval()
