"""
Test models against a list of prompts and parameters.
"""

import os
from datetime import datetime

import click
import ollama


def get_prompts(directory="prompts/"):
    """
    Recursively loads prompts from a directory tree.

    The function returns a dictionary where each key is a subdirectory name, and
    its corresponding value is another dictionary. The inner dictionary's keys are
    the file names (without extension) of prompt files found in that subdirectory,
    and their values are the contents of those files as strings.

    Args:
        directory (str): Path to the root directory containing prompts. Defaults to "prompts/".

    Returns:
        dict: A dictionary of dictionaries, where each inner dictionary contains prompts
            for a particular subdirectory.

    Examples:
        >>> get_prompts()
        {'subdir1': {'prompt1.txt': 'This is prompt 1', ...}, 
         'subdir2': {'prompt1.txt': 'This is another prompt 1', ...}}
    """

    prompts = {}

    for item in os.listdir(directory):
        path = os.path.join(directory, item)

        if os.path.isdir(path):
            prompts[item] = {}

            for file_path in os.scandir(path):
                with open(file_path, encoding="utf-8") as f:
                    # Use the file name (without extension) as the key
                    file_name = os.path.splitext(file_path.name)[0]
                    prompts[item][file_name] = f.read()

    return prompts


def run_prompt(model, prompt_test, options, test_result_file):
    """
    Run an OLLAMA prompt and generate a response.

    This function takes in a model, a prompt to test, options for the prompt,
    and a file path to save the test result. It uses the `ollama.generate`
    method to generate responses from the prompt and saves the prompts, options,
    and full response to the specified file.

    :param model: The OLLAMA model to use.
    :param prompt_test: The prompt to test.
    :param options: Options for the prompt.
    :param test_result_file: The file path to save the test result.
    """

    stream = ollama.generate(
        model=model,
        prompt=prompt_test,
        stream=True,
        options=options,
    )

    click.echo("")
    click.echo(options)

    full_response = ""
    for chunk in iter(stream):
        click.echo(chunk["response"], nl=False)
        full_response = full_response + chunk["response"]

    with open(test_result_file, "w", encoding="utf-8") as f:
        f.write(
            f"# Prompt\n"
            f"{prompt_test}\n\n"
            f"## Options\n"
            f"{options}\n\n"
            f"# Response\n"
            f"{full_response}\n"
        )


@click.command()
@click.option("--model", default="llama3.1", help="Model to test")
@click.option("--group", default=None, help="Test group to run")
@click.option("--temp", default=1, type=float, help="Temperature")
@click.option("--temp_min", default=None, type=float, help="Temperature min")
@click.option("--temp_max", default=None, type=float, help="Temperature max")
@click.option(
    "--temp_inc",
    default=0.1,
    help="How much should the temperature increase between min and max",
)
@click.option("--seed", default=42, help="Random seed")
@click.option("--num_predict", default=None, help="Max tokens to generate")
def start(model, group, temp, temp_min, temp_max, temp_inc, seed, num_predict):
    """
    Start the tests with the specified parameters
    """
    click.echo("Selected options:")
    click.echo(f"  Model: {model}")
    click.echo(f"  Random seed: {seed}")
    click.echo(f"  Max tokens: {num_predict}")
    if temp_min is not None and temp_max is not None:
        temperatures = []
        t = temp_min
        while t <= temp_max:
            temperatures.append(round(t, 2))
            t = t + temp_inc
        click.echo(f"  Temperatures: {temperatures}")
    else:
        temperatures = [temp]

    # Read prompts
    prompts = get_prompts()

    current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    results_directory = f"results_{current_timestamp}"
    os.mkdir(results_directory)

    for group_name, group_prompt_test in prompts.items():
        # Filter group test
        if group is not None and group_name != group:
            continue

        test_group_directory = results_directory + "/" + group_name
        os.mkdir(test_group_directory)

        for prompt_test, prompt_contents in group_prompt_test.items():
            click.echo(f"Prompt: {prompt_contents}")

            for temperature in temperatures:
                options = {
                    "temperature": temperature,
                    "seed": seed,
                    "num_predict": num_predict,
                }

                test_result_file = (
                    f"{test_group_directory}/{prompt_test}_temp={temperature}"
                )

                run_prompt(
                    model=model,
                    prompt_test=prompt_contents,
                    options=options,
                    test_result_file=test_result_file,
                )

            click.echo("")
            click.echo("---")
            click.echo("")


if __name__ == "__main__":
    start()
