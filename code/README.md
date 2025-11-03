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

## Development

### Code Quality

This project uses several tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting

Run code quality checks:
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/
```

## CI/CD

This project uses GitHub Actions for continuous integration. The CI pipeline includes:

### Automated Checks

- **Multi-Python Version Testing**: Tests run on Python 3.9, 3.10, and 3.11
- **Code Coverage**: Generates coverage reports and uploads to Codecov
- **Linting**: Flake8 checks for code quality issues
- **Code Formatting**: Ensures consistent formatting with Black
- **Import Sorting**: Validates import organization with isort
- **Docker Build**: Verifies Docker image builds successfully

### CI Triggers

The CI pipeline runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

### Local CI Simulation

To run the same checks locally:

```bash
# Install development dependencies
pip install pytest pytest-cov flake8 black isort

# Run tests with coverage
pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing

# Run linting
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Check formatting
black --check --diff src/ tests/
isort --check-only --diff src/ tests/

# Build Docker image
docker build -t myocardial-blood-flow .
```

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass and code quality checks pass
6. Submit a pull request

## License

[Add your license information here]
