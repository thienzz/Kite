# Contributing to Kite Framework

First off, thank you for considering contributing to Kite! 🎉

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When you create a bug report, include:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, error messages)
- **Describe the behavior you observed** and what you expected
- **Include your environment details** (Python version, OS, Kite version)

Example bug report:
```markdown
**Title**: Circuit breaker not opening after 3 failures

**Description**: 
Circuit breaker is configured with failure_threshold=3 but doesn't 
open even after 5 consecutive failures.

**Steps to Reproduce**:
1. Configure circuit breaker with threshold=3
2. Cause 5 consecutive LLM call failures
3. Observe circuit breaker state

**Expected**: State should be OPEN after 3 failures
**Actual**: State remains CLOSED

**Environment**:
- Kite version: 0.1.0
- Python: 3.10.0
- OS: Ubuntu 22.04
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the proposed functionality
- **Explain why this enhancement would be useful**
- **Provide examples** of how it would be used

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Add tests** if you've added code that should be tested
3. **Update documentation** if you've changed APIs
4. **Ensure tests pass**: `pytest tests/`
5. **Follow the code style**: `black kite/ && flake8 kite/`
6. **Write a clear commit message**

Example workflow:
```bash
# Fork and clone
git clone https://github.com/thienzz/Kite.git
cd Kite

# Create branch
git checkout -b feature/my-new-feature

# Make changes
# ... code ...

# Test
pytest tests/

# Format
black kite/
flake8 kite/

# Commit
git add .
git commit -m "Add feature: description"

# Push
git push origin feature/my-new-feature

# Open Pull Request on GitHub
```

## Development Setup
```bash
# Clone repository
git clone https://github.com/thienzz/Kite.git
cd Kite

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"  # Install in editable mode with dev dependencies

# Run tests
pytest tests/

# Run examples
python examples/case1_ecommerce_support.py
```

## Code Style

- Follow PEP 8
- Use `black` for formatting: `black kite/`
- Use `flake8` for linting: `flake8 kite/`
- Use type hints where possible
- Write docstrings for public APIs

Example:
```python
def create_agent(
    self, 
    name: str, 
    system_prompt: str = "", 
    tools: Optional[List[Tool]] = None,
    **kwargs
) -> Agent:
    """
    Create a general-purpose agent.
    
    Args:
        name: Agent name
        system_prompt: System prompt for the agent
        tools: List of tools available to the agent
        **kwargs: Additional configuration
        
    Returns:
        Agent instance with configured tools
        
    Example:
        >>> agent = ai.create_agent(
        ...     name="Assistant",
        ...     system_prompt="You are helpful",
        ...     tools=[search_tool]
        ... )
    """
    # Implementation
```

## Testing

- Write unit tests for new features
- Ensure existing tests pass
- Aim for >70% code coverage
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=kite tests/

# Run specific test
pytest tests/test_circuit_breaker.py
```

## Documentation

- Update README.md if adding major features
- Update docs/ for API changes
- Add examples for new functionality
- Keep CHANGELOG.md updated

## Questions?

Feel free to open an issue with the "question" label or reach out to maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.