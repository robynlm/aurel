Thank you for your interest in contributing to aurel! This document provides guidelines for contributing to the project.

# Reporting Issues

Found a bug or have a suggestion? Please open an issue:
- [Bug report](https://github.com/robynlm/aurel/issues/new?template=bug_report.md)
- [Feature request](https://github.com/robynlm/aurel/issues/new?template=feature_request.md)
- [Question](https://github.com/robynlm/aurel/issues/new?template=question.md)

Or feel free to directly contact Robyn Munoz at <r.l.munoz@sussex.ac.uk>

# Code Contributions

Before you start coding:
- Check [existing issues](https://github.com/robynlm/aurel/issues) and [pull requests](https://github.com/robynlm/aurel/pulls) to avoid duplication
- For major changes, [open an issue](https://github.com/robynlm/aurel/issues/new/choose) first to discuss your approach

## Branch Workflow

This project follows a two-branch workflow:
- **`main`**: Stable release branch - protected and only updated via reviewed PRs from `development`
- **`development`**: Active development branch - all contributions should target this branch

**Important**: When submitting a pull request, make sure it targets the `development` branch, not `main`.

**Guidelines for code contributions:**
- Follow [PEP 8 style](https://peps.python.org/pep-0008/)
- Write clear docstrings for functions and classes using [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
- Include tests for new features in the `tests/` directory (use `test_*.py` filenames and `test_*` function names)
- Update documentation (docstrings, `.rst` files in `docs/source/`, or add notebooks in `docs/notebooks/`)
- [Make sure everything works](check) before submitting

## Getting Started

- **Fork the repository on GitHub**
   - Log in to your GitHub account at YOUR_USERNAME
   - Go to the [aurel repository](https://github.com/robynlm/aurel) and click on the "Fork" button
   
   This creates your own copy at `github.com/YOUR_USERNAME/aurel`. You'll get all branches (including `main` and `development`).

- **Clone your fork locally:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/aurel.git
   cd aurel
   ```

- **Set up the upstream remote** (to track the original repository):
   ```bash
   git remote add upstream https://github.com/robynlm/aurel.git
   ```

- **Create a branch for your changes:**
   
   Start from `development`, not `main`:
   ```bash
   git checkout development
   git checkout -b your-feature-name
   ```
   
   **Why create a branch?** Even though it's your fork, using a branch:
   - Keeps your `development` branch clean and synced with the original
   - Lets you work on multiple features simultaneously
   - Makes it easier to update if the original repository changes
   - Clearly identifies what your PR is about

## Development Setup

From the root directory of the repository:

- **Install in development mode with all dependencies:**
   ```bash
   pip install -e .[test,docs]
   ```
   This installs the package in editable mode with optional test and documentation dependencies defined in `pyproject.toml`.

(check)=
- **Check everything works**:

   - **Run tests:**
      ```bash
      pytest
      ```

   - **Build documentation:**
      ```bash
      cd docs
      make clean html
      ```
      
      To view the built documentation locally:
      ```bash
      cd _build/html
      python3 -m http.server
      ```
      Then open [http://localhost:8000](http://localhost:8000) in your browser. Check that your documentation changes appear correctly and there are no formatting issues or broken links.

(keeping-your-fork-updated)=
## Keeping Your Fork Updated

While you're working on your changes, the original repository may get new commits. To stay in sync:

- **Fetch updates from the original repository:**
   ```bash
   git fetch upstream
   ```
   This downloads new commits from the original repo but doesn't change your files yet.

(update-development)=
- **Update your local development branch:**
   ```bash
   git checkout development
   git merge upstream/development
   ```
   This updates your local `development` branch with the latest changes from the original repo.
   If there are updates present here, follow the rest of this section to bring these updates over to your feature branch and update your fork. 
   Otherwise you're good to go and continue your work.

- **Update your feature branch with those new changes:**
   
   Now you want to include those new `development` commits into your `your-feature-name` branch. You have two main options:
   
   - **Option A: Merge** (simpler) - If you already pushed your branch to GitHub, use merge. This adds the new commits from `development` into your feature branch and creates an extra merge commit showing when you incorporated the updates.
      ```bash
      git checkout your-feature-name
      git merge development
      ```
   
   - **Option B: Rebase** (cleaner history, more advanced) - If you haven't pushed yet, or don't mind force-pushing, use rebase. This replays your changes on top of the latest `development`, as if you started working from there. No extra merge commit.
      
      ```bash
      git checkout your-feature-name
      git rebase development
      ```
   
   - **Handling Conflicts**
      
      If git says there are conflicts, it means you and someone else changed the same lines of code. Git can't decide which version to keep, so you need to manually resolve it:
      
      - Git will mark the conflicting files. Open them and look for conflict markers like `<<<<<<< HEAD`, `=======`, and `>>>>>>>`.
      
      - Edit the file to keep what you want (delete the markers and choose/combine the code).
      
      - Mark the file as resolved, add it, and continue with the merge or rebase.
      
         ```bash
         git add filename.py
         git merge --continue
         # OR
         git rebase --continue
         ```
      
      - If conflicts are too difficult, you can cancel and return to how things were before you started the merge/rebase.
      
         ```bash
         git merge --abort
         # OR
         git rebase --abort
         ```
   
   - **Alternative Approach: Start Fresh**
   
      After you've [updated your `development` branch](update-development), you can create a new feature branch from it. Review what changes you made in your old branch with `git diff`, then manually apply your changes to the new branch. For specific files, you can copy them from the old branch. This way you start with a clean slate and can carefully reapply your changes without conflicts.
   
      ```bash
      git checkout development
      git checkout -b your-feature-name-v2
      git diff your-feature-name-v2 your-feature-name    # Review changes
      git checkout your-feature-name -- path/to/file.py  # Copy specific files
      ```
      
      Then continue working on `your-feature-name-v2`. You can delete the old branch later with `git branch -D your-feature-name` once you're sure you've copied everything you need.

- **Push updates to your fork**
   
   Update your fork's development and feature branch. If you used merge, push your feature branch normally. If you used rebase, you need `--force` because you've rewritten history.
   
   ```bash
   git push origin development
   git push origin your-feature-name          # If you used merge
   # OR
   git push --force origin your-feature-name  # If you used rebase
   ```

## Submitting Changes

Once you've completed your work on your feature branch and committed and pushed all your changes to your fork:

- **Make sure you're up to date with the latest development branch:**
   
   This should have been regularly checked throughout your work, but verify one last time before submission. Follow the steps in [Keeping Your Fork Updated](keeping-your-fork-updated) to sync with the latest changes.

- [**Make sure everything works**](check)

- **Create a Pull Request on GitHub:**
   - Go to your fork on GitHub
   - Click "Pull Request" and select `development` as the base branch (not `main`)
   - Fill out the [pull request template](https://github.com/robynlm/aurel/blob/main/.github/PULL_REQUEST_TEMPLATE.md) that will automatically appear
   
A maintainer will review your PR and may request changes before merging.

# License

By contributing to aurel, you agree that your contributions will be licensed under the [GNU General Public License v3.0](https://github.com/robynlm/aurel/blob/main/LICENSE).

Thank you for helping make aurel better!
