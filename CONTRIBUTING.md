# Contributing to GraphEm Rapids

Thank you for your interest in contributing to GraphEm Rapids!

## Quick Start

1. **Fork & Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/graphem-rapids.git
   cd graphem-rapids
   ```

2. **Setup Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -e ".[test,docs]"
   ```

3. **Run Tests**
   ```bash
   pytest tests/ -v
   python examples/example_usage.py  # Test examples work
   ```

## Development Guidelines

### Code Style
- Follow **PEP 8**
- Add **type hints** to all public functions
- Use **NumPy-style docstrings**
- Test your changes with `pytest tests/`

### Making Changes
- Create a feature branch: `git checkout -b feature/your-feature`
- Make focused commits with clear messages
- Add tests for new functionality
- Update documentation if needed

### Pull Request Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated (if needed)
- [ ] Examples still work
- [ ] Clear PR description with what/why/how

## Need Help?

- **Report bugs**: [GitHub Issues](https://github.com/sashakolpakov/graphem-rapids/issues)
- **Suggest features**: [GitHub Discussions](https://github.com/sashakolpakov/graphem-rapids/discussions)

## Types of Contributions

- **Bug fixes** - Always welcome!
- **New graph generators** - Expand the collection
- **Performance improvements** - PyTorch/RAPIDS optimization
- **Backend enhancements** - cuVS integration, memory management
- **Documentation** - Examples, tutorials, API docs
- **Tests** - Improve coverage and reliability

---
**Questions?** Open an issue or start a discussion.
