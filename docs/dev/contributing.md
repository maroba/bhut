# Contributing

We welcome contributions to bhut! This guide will help you set up a development environment and understand our development workflow.

## Development Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/username/bhut.git
cd bhut

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"
```

### 2. Install Development Tools

```bash
# Install pre-commit hooks for automatic formatting
pre-commit install
```

## Running Tests

### Full Test Suite

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=bhut --cov-report=html
```

### Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/

# Edge cases
pytest tests/edge_cases/
```

## Code Quality

### Formatting and Linting

We use **ruff** for both formatting and linting:

```bash
# Format code
ruff format .

# Check and fix linting issues
ruff check . --fix

# Check without fixing
ruff check .
```

### Type Checking

We use **mypy** for static type checking:

```bash
# Run type checking
mypy bhut/

# Run with strict mode
mypy --strict bhut/
```

## Pull Request Guidelines

### Before Submitting

1. **Test your changes**: Ensure all tests pass
2. **Add tests**: For new features or bug fixes
3. **Update docs**: If you change the public API
4. **Check types**: Run mypy without errors
5. **Format code**: Run ruff format

### PR Checklist

- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Documentation updated if needed
- [ ] Type hints added for new code
- [ ] Descriptive commit messages
- [ ] No unnecessary dependencies added

### Commit Message Format

```
type: short description

Longer explanation if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

## Development Tips

- Use descriptive variable names and add type hints
- Keep functions small and focused
- Add docstrings for public API
- Consider performance implications for hot paths
- Test edge cases (empty arrays, single particles, etc.)

## Getting Help

- Open an issue for bugs or feature requests
- Start discussions for design questions
- Check existing issues before creating new ones