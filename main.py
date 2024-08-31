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
@click.option("--seed", default=42, help="Random seed")
@click.option("--num_predict", default=None, help="Max tokens to generate")
# Temperature
@click.option("--temp", default=1, type=float, help="Temperature")
@click.option("--temp_min", default=None, type=float, help="Temperature min")
@click.option("--temp_max", default=None, type=float, help="Temperature max")
@click.option(
    "--temp_inc",
    default=0.1,
    help="How much should the temperature increase between min and max",
)
# Top_k
@click.option("--top_k", default=1, help="Number of top scoring predictions to display")
@click.option(
    "--top_k_min",
    default=None,
    type=int,
    help="Minimum number of top K predictions to consider",
)
@click.option(
    "--top_k_max",
    default=None,
    type=int,
    help="Maximum number of top K predictions to consider",
)
@click.option(
    "--top_k_inc",
    default=1,
    help="How many should the top K increase between min and max",
)
def start(
    model,
    group,
    seed,
    num_predict,
    # Temperature
    temp,
    temp_min,
    temp_max,
    temp_inc,
    # Top_k
    top_k,
    top_k_min,
    top_k_max,
    top_k_inc,
):
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
    else:
        temperatures = [temp]
    click.echo(f"  Temperatures: {temperatures}")

    if top_k_min is not None and top_k_max is not None:
        top_k_values = []
        k = top_k_min
        while k <= top_k_max:
            top_k_values.append(k)
            k = k + top_k_inc
    else:
        top_k_values = [top_k]
    click.echo(f"  Top K values: {top_k_values}")

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

            options = {
                "seed": seed,
                "num_predict": num_predict,
            }

            # Iterate temperature
            for temperature in temperatures:
                options["temperature"] = temperature

                test_result_file = (
                    f"{test_group_directory}/{prompt_test}_temp={temperature}"
                )

                run_prompt(
                    model=model,
                    prompt_test=prompt_contents,
                    options=options,
                    test_result_file=test_result_file,
                )

            # Iterate top_k
            for top_k_value in top_k_values:
                options["top_k"] = top_k_value

                test_result_file = (
                    f"{test_group_directory}/{prompt_test}_top_k={top_k_value}"
                )

                run_prompt(
                    model=model,
                    prompt_test=prompt_contents,
                    options=options,
                    test_result_file=test_result_file,
                )

            # Iterate runtime options
            # num_predict: int
            # top_k: int
            # top_p: float
            # tfs_z: float
            # typical_p: float
            # repeat_last_n: int
            # temperature: float
            # repeat_penalty: float
            # presence_penalty: float
            # frequency_penalty: float
            # mirostat: int
            # mirostat_tau: float
            # mirostat_eta: float

            click.echo("")
            click.echo("---")
            click.echo("")


if __name__ == "__main__":
    start()
