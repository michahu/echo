# ECHO: Experience Consolidation via Hindsight Optimization

**New environments: XMiniGrid-Stateful, PeopleJoin-QA-Stateful**

✨ Check out XMiniGrid-Stateful, under `/xminigrid`, for an easy-to-run online learning benchmark for LLM agents!

This repository contains the implementation of ECHO, a prompting framework that adapts hindsight experience replay from reinforcement learning for language model agents. ECHO generates optimized trajectories for alternative goals from failed attempts, creating synthetic positive examples that improve sample efficiency in novel environments. We evaluate ECHO on XMiniGrid-Stateful and PeopleJoin-QA-Stateful, demonstrating up to 80% performance improvements over vanilla language agent baselines.

## Installation
Install [uv](https://docs.astral.sh/uv/) following its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
   - On macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - On Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - Or using pip: `pip install uv`

uv will automatically manage Python 3.11 installation. If you need to install Python 3.11 explicitly, run `uv python install 3.11`.    Then run `uv sync` to create a virtual environment and install all dependencies.

## XMiniGrid-Stateful

In XMiniGrid-Stateful, an LLM navigation agent picks up objects in a GridWorld environment. Between episodes, the agent and all objects get reset to the same positions, allowing the agent to hypothesize or recall better ways to reach objects between episodes.

To run XMiniGrid-Stateful, do the following:
```bash
cd xminigrid
bash scripts/run.sh
```

## PeopleJoin-QA-Stateful

**Note: If you encounter issues with PeopleJoin, request support at [microsoft/peoplejoin](https://github.com/microsoft/peoplejoin)**

PeopleJoin is a benchmark for evaluating LM-mediated collaborative problem solving. Given a user request, PeopleJoin agents must identify teammates who might be able to assist, converse with these teammates to gather information, and finally compile a useful answer or summary for the original user. PeopleJoin-QA-Stateful asks the agent several questions about tabular data in a row.

To run PeopleJoin-QA-Stateful, do the following:
```bash
cd peoplejoin
bash workspace/scripts/make_peoplejoinqa_S.sh
make backend
bash workspace/scripts/run_peoplejoinqa_S.sh [YOUR-PORT] none echo
bash workspace/scripts/eval_all.sh
```

### Setup

1. Activate the virtual environment:
   Run `uv shell` or use `uv run` to run commands in the virtual environment.
2. Run `uv run make backend` to start a back-end server at `http://127.0.0.1:8000/`.
    - To utilize Make file in Windows, you will have to install make on Windows. One way to do that is through chocolatey.
    - To install chocolatey, run the following command in the administrative PowerShell:
   `Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))`
    - Once installed, restart Powershell and run: `choco install make`. You can verify your installation using `make -v`.
    - You can provide config name such as `uv run make backend AGENT_CONF=src/async_collab/scenarios/people_join_qa/agent_configs/spider_sample.json`


### `peoplejoin` directory information
- `src/`: Python code for the project.
- `tests/`: Unit tests for code in `src/`.
- `data/`: Releant data files
- `workspace/`: experiments scripts
- `src/async_collab/llm/llm_client_service.py`: LLM API interface. 


## Citation
```
@misc{hu2025sampleefficient,
      title={Sample-Efficient Online Learning in LM Agents via Hindsight Trajectory Rewriting}, 
      author={Michael Y. Hu and Benjamin Van Durme and Jacob Andreas and Harsh Jhamtani},
      year={2025},
      eprint={2510.10304},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.10304}, 
}
}
```
