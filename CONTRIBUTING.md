# Contributing

Thank you for your interest in contributing to this project.

## Getting started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for all Python code.
- Add docstrings (Google style) to any new public functions or classes.
- Keep functions focused — one clear responsibility per function.

## Running a syntax check

```bash
python -m py_compile main.py pipeline.py config.py data_loader.py \
  text_processor.py sentiment_analyzer.py visualizer.py
echo "Syntax OK"
```

## Pull requests

- Open a pull request against the `main` branch.
- Include a clear description of what changed and why.
- Ensure the CI checks pass before requesting review.

## Reporting issues

Open a GitHub issue with:
- A description of the bug
- Steps to reproduce it
- Expected vs. actual behavior
