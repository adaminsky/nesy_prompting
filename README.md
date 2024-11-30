# Expert-free Neurosymbolic with Foundation Models

- Foundation models (FMs) seem to "understand" general raw data.
- How can we build systems ontop of this understanding, allowing us to "program"
using abstractions from a FM?
- For example, LLMs can perform some neurosymbolic tasks (involving input
understanding as well as reasoning) zero-shot, but certain forms of prompt
engineering such as Chain-of-Thought can significantly improve their
performance.
- We formalize this problem as neurosymbolic using foundation models, where a
foundation model can act as both a system for input understanding as well as
program synthesis, allowing us to replace the requirement for experts in
neurosymbolic with scale.

## Setup
Start by creating a virtual environment with `make create_environment`. This
uses the current version of python used when you call `python`, so you should
first create a conda environment with a specific python version if running
`python --version` is not Python 3.10+.

Activate the newly created environment with `source .venv/bin/activate`.
Finally, install the requirements with `make requirements`.


### Download Datasets
First create the `data/` directory.

#### HWF
Download from
[here](https://drive.google.com/file/d/1VW--BO_CSxzB9C7-ZpE3_hrZbXDqlMU-/view?usp=share_link)
and place under `data/HWF`.

#### CLEVR
Download from [here](https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip) and
place under `data/CLEVR_v1.0`.

#### Pathfinder

Download my version of the code and then follow the instructions in the README.

#### Mystery Blocksworld

Download from [here](https://raw.githubusercontent.com/karthikv792/LLMs-Planning/refs/heads/main/plan-bench/prompts/mystery_blocksworld/task_1_plan_generation.json).
