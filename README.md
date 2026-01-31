## Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for package management.

### Initialize Submodules

The `images/` directory is a git submodule. Before running the notebook, ensure the submodules are initialized:

```bash
git submodule update --init --recursive
```

## Running the Project

The project is implemented as a **marimo** notebook (`notebook.py`).

### Run the Notebook in Sandbox Mode

To run the notebook with all dependencies:

```bash
uvx marimo edit --sandbox notebook.py
```
