# Setup and run (Python venv)

This project requires Python 3.8+.

Quick steps (Linux / bash):

1. Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Run a quick smoke test (examples):

```bash
python test.py
# or run specific scripts, e.g.:
python test_kalman.py
```

Helper script:

There is a helper script to automate steps 1–2:

```bash
bash scripts/setup_venv.sh
```

Notes:
- Activating a venv inside a script won't persist to your interactive shell; run the `source .venv/bin/activate` command yourself when you want to work interactively.
- If you use a different Python executable (pyenv/conda), replace `python3` with the appropriate command.
