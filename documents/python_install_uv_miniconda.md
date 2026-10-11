# Install Python with uv or Miniconda

A beginner's guide for this numerical engineering course on macOS and Linux.
Choose one workflow below. You can have both tools installed, but you only need
one to run the lessons. Run commands one line at a time in Terminal.

## 1. What are you installing?

**Python** is the interpreter that executes your `.py` files. **Packages**, such
as NumPy and SciPy, add mathematical tools. An **environment** keeps a project's
Python and packages separate from other projects.

Installing Python alone does not install all the course packages. The setup has
three steps: install a manager, create an environment with Python, and install
its dependencies.

| Question | uv | Miniconda and conda |
|---|---|---|
| What is it? | A Python package and project manager that can also install Python | Miniconda is a small installer containing Python, conda, and essential packages; conda manages environments and packages |
| Main purpose | Manage Python projects, dependencies, and tools efficiently | Manage environments containing Python and other software, including native libraries |
| Package source | Python package indexes, normally PyPI | Conda channels containing conda packages |
| Typical environment | A `.venv` folder inside the project | A named environment, such as `numerical-course`, stored under the conda installation by default |
| Project specification | `pyproject.toml` and `uv.lock` | Commonly `environment.yml` |
| Run a program | `uv run python script.py` | Activate the environment, then `python script.py` |

uv installs compiled Python wheels when available, so it also works well with
NumPy and SciPy. Conda can additionally manage non-Python dependencies as
packages; it is useful when a scientific software stack requires specific native
libraries or your research group supplies a conda environment.
See the [uv overview](https://docs.astral.sh/uv/) and
[conda overview](https://docs.conda.io/projects/conda/en/stable/).

**For this course, start with uv** if you have no existing preference: the project
already includes `pyproject.toml`, `uv.lock`, and `.python-version`. Choose
Miniconda if you already use conda or need it for other scientific software.
Both approaches can run these lessons.

## 2. Option A: install uv and set up the course

### Install uv once

On macOS or Linux:

```sh
# Download and run the official uv installer.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen Terminal so the installer's PATH changes take effect. Check:

```sh
uv --version
command -v uv
```

Alternatively, on macOS with Homebrew already installed, use `brew install uv`.
Choose one installation method. uv itself does not require an existing Python
installation. Details: [uv installation](https://docs.astral.sh/uv/getting-started/installation/).

### Install Python and the course dependencies

Enter the course's root folder, where `pyproject.toml` is located. Replace the
sample path with the folder where you extracted or cloned the course:

```sh
# replace the <path/to/numerical_engineering_course> with the exact path of your project.
cd "<path/to/numerical_engineering_course>"

# Install the latest available patch release of Python 3.12.
uv python install 3.12

# Create/update .venv using the course's existing lockfile.
uv sync --locked

# Verify the interpreter and mathematical packages.
uv run python --version
uv run python -c "import sys; print(sys.executable)"
uv run python -c "import numpy, scipy, matplotlib; print('Course packages ready')"

# Open the notebooks in JupyterLab.
uv run jupyter lab
```

`uv python install` installs an interpreter managed by uv. It does not install
NumPy or the other course packages. uv can also download a missing Python
version automatically. See [installing Python with uv](https://docs.astral.sh/uv/guides/install-python/).

`uv sync --locked` installs the dependencies selected by `uv.lock` and checks
that the lockfile agrees with `pyproject.toml`. `.python-version` selects 3.12
for this course. `uv run` runs commands in the project environment; activation
is unnecessary. See [uv project workflow](https://docs.astral.sh/uv/guides/projects/).

For later sessions, return to the course folder and run `uv run jupyter lab`.
To run the first lesson as a script:

```sh
uv run python scripts/01_vectors_matrices.py
```

### Optional: activate the environment manually

```sh
source .venv/bin/activate
python --version
python scripts/01_vectors_matrices.py
deactivate
```

Activation makes this environment's Python available as `python` in the current
shell. `deactivate` restores the previous shell environment; it does not delete
packages. See [uv environments](https://docs.astral.sh/uv/pip/environments/).

## 3. Option B: install Miniconda and set up the course

### Install Miniconda once

Download the installer for your operating system and processor from the
[official Miniconda installer directory](https://repo.anaconda.com/miniconda/).
You can check the processor reported by your terminal with:

```sh
uname -m
```

| Computer | Shell installer to select |
|---|---|
| Apple Silicon Mac (`arm64`) | `Miniconda3-latest-MacOSX-arm64.sh` |
| Intel Mac (`x86_64`) | A supported `MacOSX-x86_64.sh` installer from the archive |
| Linux Intel/AMD 64-bit (`x86_64`) | `Miniconda3-latest-Linux-x86_64.sh` |
| Linux ARM 64-bit (`aarch64`) | `Miniconda3-latest-Linux-aarch64.sh` |

Installer availability and operating-system requirements can change. For an
Intel Mac, check the supported archived release rather than assuming the newest
release supports it. A terminal running through Rosetta can report `x86_64` on
an Apple Silicon Mac; prefer a native terminal and the ARM installer there.

For example, after downloading the Apple Silicon shell installer:

```sh
bash "$HOME/Downloads/Miniconda3-latest-MacOSX-arm64.sh"
```

On Linux, replace the filename with the Linux installer you downloaded. Follow
the prompts, review the license, choose a location such as `~/miniconda3`, and
allow shell initialization when prompted. Reopen Terminal afterward.
See the official [macOS](https://docs.conda.io/projects/conda/en/stable/user-guide/install/macos.html)
and [Linux](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)
installation guides.

```sh
conda --version
conda info --envs
```

### Create a dedicated course environment

```sh
# Create a named environment with Python and the course packages.
conda create -n numerical-course python=3.12 "numpy>=1.24,<3" "scipy>=1.10,<2" "matplotlib>=3.7,<4" "jupyterlab>=4,<5"
conda activate numerical-course

# Enter the course folder; replace this sample path.
cd "$HOME/path/to/numerical_engineering_course"
python --version
python -c "import sys; print(sys.executable)"
python -c "import numpy, scipy, matplotlib; print('Course packages ready')"
jupyter lab
```

The quoted package ranges match this course's dependency declarations. This
conda workflow resolves conda packages independently of `uv.lock`, so exact
versions and builds can differ from the uv environment.

For later sessions:

```sh
conda activate numerical-course
cd "$HOME/path/to/numerical_engineering_course"
python scripts/01_vectors_matrices.py
# Or open the notebooks:
jupyter lab
```

Leave the environment with `conda deactivate`. Keep course packages in the
named environment rather than `base`, which contains conda's own tools.

To save and recreate a shareable environment specification:

```sh
conda env export --from-history > environment.yml
# On another computer, from the directory containing environment.yml:
conda env create -f environment.yml
```

`--from-history` records explicitly requested packages; it does not lock every
resolved dependency or guarantee identical builds. Details:
[conda environment management](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).

## 4. Check which Python you are running

An environment name in the prompt is helpful, but the interpreter path is the
clearest check. For uv, use `uv run python` from the course folder. For conda,
activate `numerical-course` first, then use `python`:

```sh
python -c "import sys; print(sys.executable)"
command -v python
which python
```

For uv, expect a path ending in `.venv/bin/python`. For conda, expect a path
containing `envs/numerical-course/bin/python` with a default installation layout.
See [shell basics](shell_git_github_basics.md) for `which`, `where`, and `PATH`.

In VS Code, select the same environment as both the Python interpreter and the
notebook kernel. Inside a notebook, run this cell to check:

```python
import sys
print(sys.executable)
```

## 5. If both managers are installed

Use one workflow per course environment. Before activating `.venv`, leave an
active conda environment with `conda deactivate`; before activating conda, leave
an active virtual environment with `deactivate`.

`uv run` normally uses the project's `.venv`, while `uv pip install` can target
an active conda environment. Avoid mixing those commands while learning, since
you can install into a different environment than the one running the notebook.
See [uv environment discovery](https://docs.astral.sh/uv/pip/environments/).

## 6. Common problems

| Problem | What to check |
|---|---|
| `uv: command not found` | Reopen Terminal and follow the PATH instructions printed by the uv installer |
| `conda: command not found` | Confirm installation and initialize the correct shell using the installed conda executable |
| `conda activate` asks for initialization | With a default install, run `~/miniconda3/bin/conda init zsh` for Zsh or `~/miniconda3/bin/conda init bash` for Bash, then reopen Terminal; adjust the installation path if needed |
| `ModuleNotFoundError: numpy` | Check `sys.executable`; rerun `uv sync --locked` or activate the conda course environment |
| `uv sync --locked` reports an outdated or missing lockfile | Check that you have the complete course download; deliberate dependency changes require regenerating the lockfile |
| Terminal works but notebook imports fail | Select the matching notebook kernel and restart it |

This document describes installation commands; creating it does not install or
change Python on your computer.
