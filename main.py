import os
from datetime import datetime

import click
import ollama


def get_prompts(directory="prompts/"):
    prompts = {}

    for item in os.listdir(directory):
        path = os.path.join(directory, item)

        if os.path.isdir(path):
            prompts[item] = []

            for file_path in os.scandir(path):
                with open(file_path, encoding="utf-8") as f:
                    prompts[item].append(f.read())

    return prompts


@click.command()
@click.option("--model", default="llama3.1", help="Model to test")
@click.option("--temp", default=1, type=float, help="Temperature")
@click.option("--temp_min", default=None, help="Temperature min")
@click.option("--temp_max", default=None, help="Temperature max")
@click.option("--seed", default=42, help="Random seed")
@click.option("--num_predict", default=2048, help="Max tokens to generate")
@click.option(
    "--temp_inc",
    default=0.1,
    help="How much should the temperature increase between min and max",
)
def start(model, temp, temp_min, temp_max, temp_inc, seed, num_predict):
    """
    Start the tests with the specified parameters
    """
    click.echo("Selected options:")
    click.echo(f"  Model: {model}")
    click.echo(f"  Random seed: {seed}")
    click.echo(f"  Max tokens: {num_predict}")
    click.echo(
        f"  Temperature: {temp} (range: {temp_min if temp_min else '-'}/{temp_max if temp_max else '-'})"
    )
    if temp_min is not None and temp_max is not None:
        click.echo(f"  Temperature increment: {temp_inc}")
    # Confirm
    # click.confirm("Do you want to continue?", abort=True)

    # Read prompts
    prompts = get_prompts()

    current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    results_directory = f"results_{current_timestamp}"
    os.mkdir(results_directory)
    for test_group, group_prompt_test in prompts.items():
        test_group_directory = results_directory + "/" + test_group
        os.mkdir(test_group_directory)

        index = 0
        for prompt_test in group_prompt_test:
            index = index + 1
            click.echo(f"Prompt: {prompt_test}")

            stream = ollama.generate(
                model=model,
                prompt=prompt_test,
                stream=True,
                options={
                    "temperature": temp,
                    "seed": seed,
                    "num_predict": num_predict,
                },
            )

            full_response = ""
            for chunk in iter(stream):
                print(chunk["response"], end="", flush=True)
                full_response = full_response + chunk["response"]

            test_result_file = test_group_directory + "/" + str(index)
            with open(test_result_file, "w", encoding="utf-8") as f:
                f.write(
                    f"# Prompt\n"
                    f"{prompt_test}\n\n"
                    f"# Response\n"
                    f"{full_response}\n"
                )

            click.echo("")
            click.echo("---")
            click.echo("")


if __name__ == "__main__":
    start()

