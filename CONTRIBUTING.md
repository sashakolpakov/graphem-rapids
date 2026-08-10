# Contributing to GraphEm RAPIDS

Thank you for contributing to GraphEm RAPIDS. Keep changes focused and preserve
the canonical algorithm and its fail-closed diagnostics.

## Development environment

1. Fork and clone the repository.

   ```bash
   git clone https://github.com/YOUR_USERNAME/graphem-rapids.git
   cd graphem-rapids
   ```

2. Create a Python 3.11 environment and install the CUDA-matched Torch wheel
   before the editable package.

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129
   python -m pip install -e ".[test,docs]"
   ```

3. Run the local checks that apply to the change.

   ```bash
   python -m pytest -q
   python -m pylint graphem_rapids tests
   python -m py_compile graphem_rapids/embedder.py
   sphinx-build -W --keep-going -n -b html docs /tmp/graphem-docs
   npx --yes markdownlint-cli2@0.23.2 "*.md"
   ```

Documentation-only CI installs the pinned CPU Torch wheel and the package with
`--no-deps` solely so autodoc can import the shared tensor API. That environment
does not qualify or execute the CUDA layout.

## Development guidelines

- Follow PEP 8 and add type hints to public functions.
- Use NumPy-style docstrings.
- Add focused tests for new behavior and failure paths.
- Do not add a second spectral solver, layout engine, score orientation, or
  silent CUDA downgrade.
- Keep production and benchmark examples pinned to `device="cuda"`.
- Update the documentation and changelog when the public contract changes.

## Pull request checklist

- [ ] The change is focused and the public behavior is explained.
- [ ] Fast tests pass locally.
- [ ] GPU-dependent claims have a separately identified GPU qualification.
- [ ] Sphinx builds with warnings treated as errors.
- [ ] Markdown lint and link checks pass.
- [ ] Examples use the current canonical API.
- [ ] Numerical-source changes include targeted correctness tests.

## Getting help

Report bugs and propose enhancements through
[GitHub Issues](https://github.com/sashakolpakov/graphem-rapids/issues).
