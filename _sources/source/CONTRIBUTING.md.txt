Thank you for your interest in contributing to aurel!

# Reporting Issues

Found a bug or have a feature request? [Open an issue](https://github.com/robynlm/aurel/issues/new/choose) or contact [r.l.munoz@sussex.ac.uk](mailto:r.l.munoz@sussex.ac.uk).

# Contributing Workflow

1. **Fork and clone:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/aurel.git
   cd aurel
   git checkout development
   git checkout -b your-feature-name
   ```

2. **Install and develop:**
   ```bash
   pip install -e ".[test,docs]"
   # Make your changes, add tests, update docs
   ```

3. **Test and check:**
   ```bash
   # Check code style
   ruff check . --respect-gitignore
   # Clean old builds to avoid conflicts
   rm -rf dist
   # Build the package
   python3 -m build
   # Install from the built wheel to test packaging
   pip install --force-reinstall --no-deps dist/*.whl
   # Run tests
   pytest
   # Build and view documentation locally
   cd docs && make clean html && python3 -m http.server --directory _build/html
   # Open http://localhost:8000
   ```

4. **Submit a pull request:**
   - Sync with latest changes
   - Commit and push
   - Create PR on GitHub targeting `development` branch (not `main`)
   - Fill out the PR template

**Note:** GitHub Actions will automatically run tests and linting on your PR. You can test locally first to catch issues early.

# License

By contributing to aurel, you agree that your contributions will be licensed under the [GNU General Public License v3.0](https://github.com/robynlm/aurel/blob/main/LICENSE).

Thank you for helping make aurel better!
