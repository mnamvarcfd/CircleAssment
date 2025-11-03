# Myocardial Blood Flow Analysis

A Python-based application for analyzing myocardial blood flow from medical imaging data (DICOM files).

## Features

- DICOM image processing and analysis
- Myocardial blood flow calculation
- Arterial input function extraction
- Comprehensive test suite with unit and integration tests
- Docker containerization support

## Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd myocardial-blood-flow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Docker Usage

Build and run with Docker:
```bash
# Build the image
docker build -t myocardial-blood-flow .

# Run the container
docker run myocardial-blood-flow
```

## Documentation

This project uses [Sphinx](https://www.sphinx-doc.org/) to generate comprehensive documentation from docstrings and RST files.

### Building Documentation Locally

To build and preview documentation locally:

```bash
# Install dependencies (includes Sphinx)
pip install -r requirements.txt

# Build HTML documentation
python -m sphinx -b html docs docs/_build/html

# Open in browser:
docs/_build/html/index.html
```

### Updating Documentation

To update documentation when making code changes:

1. **Add/update docstrings** in your Python files following NumPy/Google style:
   ```python
   def my_function(param1: int, param2: str) -> dict:
       """
       Description of what this function does.

       Args:
           param1 (int): Description of param1
           param2 (str): Description of param2

       Returns:
           dict: Description of return value
       """
       # Your implementation
       pass
   ```


#### Adding New Modules
When adding new Python modules:

1. Create the module file in `src/`
2. Add proper docstrings
3. Create corresponding RST file in `docs/api/`
4. Update `docs/index.rst` to include the new module

### Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── modules.rst          # Module overview
├── api/                 # API reference files
│   ├── data_loader.rst
│   ├── compute_quantity.rst
│   └── ...
└── _build/html/         # Generated HTML (auto-created)
```

## Development

## CI/CD

This project uses GitHub Actions for continuous integration. The CI pipeline includes:


## Project Structure

```
.
├── src/                    # Source code
│   ├── compute_quantity.py
│   ├── data_loader.py
│   ├── logging_config.py
│   ├── myocardial_blood_flow.py
│   └── save_data_manager.py
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── conftest.py        # Test configuration
│   └── test_utils.py      # Test utilities
├── input_data/            # Sample input data
├── results/               # Output results
├── .github/workflows/     # GitHub Actions CI
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
├── setup.cfg             # Project configuration
└── README.md             # This file
```

