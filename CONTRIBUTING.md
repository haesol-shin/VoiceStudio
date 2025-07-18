# Contributing to T2A-LoRA

We welcome contributions to T2A-LoRA! This document provides guidelines for contributing to the project.

## Getting Started

### Development Environment Setup

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/T2A-LoRA.git
   cd T2A-LoRA
   ```

3. Set up development environment:
   ```bash
   uv pip install -e ".[dev,docs,audio]"
   pre-commit install
   ```

### Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test them:
   ```bash
   pytest tests/ -v
   black --check src/ tests/
   flake8 src/ tests/
   ```

3. Commit your changes:
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

4. Push to your fork and create a pull request

## Code Style

We use the following tools for code formatting and linting:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking

Run formatting and checks:
```bash
# Format code
black src/ tests/ examples/
isort src/ tests/ examples/

# Check code style  
flake8 src/ tests/ examples/
mypy src/

# Run tests
pytest tests/ -v --cov=src/t2a_lora
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src/t2a_lora --cov-report=html
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Follow the existing test structure
- Aim for >90% code coverage

Example test structure:
```python
class TestYourFeature:
    """Test your feature."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        input_data = create_test_input()
        
        # Act
        result = your_function(input_data)
        
        # Assert
        assert result is not None
        assert result.shape == expected_shape
```

## Documentation

### Code Documentation

- Use clear, descriptive docstrings for all public functions and classes
- Follow Google-style docstrings:

```python
def your_function(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is invalid
    """
    pass
```

### README and Documentation Updates

- Update README.md if you add new features
- Add examples for new functionality
- Update configuration documentation if needed

## Submitting Changes

### Pull Request Process

1. **Before submitting:**
   - Ensure all tests pass: `pytest tests/`
   - Check code style: `black --check src/ && flake8 src/`
   - Update documentation if needed
   - Add tests for new functionality

2. **Pull Request Description:**
   - Clearly describe what your PR does
   - Include any breaking changes
   - Reference related issues
   - Add screenshots/examples if relevant

3. **PR Title Format:**
   - `feat: add new feature description`
   - `fix: fix bug description`
   - `docs: update documentation`
   - `test: add tests for feature`
   - `refactor: improve code structure`

### Review Process

- All PRs require at least one review
- Address review feedback promptly
- Keep PRs focused and reasonably sized
- Squash commits before merging if requested

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Python version and OS
- T2A-LoRA version
- Minimal code example that reproduces the bug
- Full error traceback
- Expected vs actual behavior

### Feature Requests

For new features:
- Describe the use case and motivation
- Provide examples of how it would be used
- Consider if it fits the project scope
- Discuss implementation approach if you have ideas

### Code Contributions

Priority areas for contributions:
- **Performance improvements**: Optimize model inference speed
- **New fusion methods**: Implement alternative multimodal fusion approaches
- **Additional TTS integration**: Support for more TTS architectures
- **Data augmentation**: Implement training data augmentation techniques
- **Evaluation metrics**: Add new evaluation methods
- **Documentation**: Improve guides and examples

## Development Guidelines

### Code Organization

- Keep modules focused and cohesive
- Use clear, descriptive names
- Follow existing project structure
- Add appropriate type hints

### Model Development

- Test new models with small configurations first
- Provide configuration examples
- Document model parameters and their effects
- Include evaluation on standard benchmarks

### Performance Considerations

- Profile code for performance bottlenecks
- Use appropriate tensor operations for GPU acceleration
- Consider memory usage for large models
- Add benchmarking for new features

## Community Guidelines

### Communication

- Be respectful and constructive in discussions
- Ask questions if anything is unclear
- Help other contributors when possible
- Use GitHub issues and discussions for project-related communication

### Code of Conduct

- Follow the project's code of conduct
- Be inclusive and welcoming to all contributors
- Focus on constructive feedback
- Respect different perspectives and approaches

## Release Process

### Version Numbering

We follow semantic versioning (SemVer):
- Major version: Breaking changes
- Minor version: New features (backward compatible)
- Patch version: Bug fixes

### Changelog

- Update CHANGELOG.md for significant changes
- Include migration notes for breaking changes
- Credit contributors for their work

## Getting Help

If you need help with development:

1. Check existing documentation and examples
2. Search GitHub issues for similar questions
3. Create a new issue with the "question" label
4. Join our Discord/Slack for real-time discussion (if available)

## Recognition

Contributors will be:
- Listed in the README.md contributors section
- Credited in release notes for significant contributions
- Added to the AUTHORS file for major contributions

Thank you for contributing to T2A-LoRA! Your contributions help make this project better for everyone.
