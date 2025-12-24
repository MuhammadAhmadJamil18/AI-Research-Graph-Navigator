# Code Quality Setup Guide

This guide explains how to set up and use the code quality tools that have been configured for this project.

## Tools Installed

1. **Black** - Code formatter
2. **isort** - Import sorter
3. **flake8** - Linter
4. **mypy** - Type checker
5. **pre-commit** - Git hooks framework

## Quick Setup

### 1. Install Development Dependencies

```bash
pip install black isort flake8 mypy pre-commit
```

Or install from requirements.txt (development tools are included):
```bash
pip install -r requirements.txt
```

### 2. Set Up Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Test the hooks (optional)
pre-commit run --all-files
```

This will automatically run code quality checks before each commit.

## Manual Usage

### Format Code with Black

```bash
# Format all Python files
black .

# Format specific file
black src/rag_pipeline.py

# Check what would be changed (dry run)
black --check .
```

### Sort Imports with isort

```bash
# Sort imports in all files
isort .

# Sort imports in specific file
isort src/rag_pipeline.py

# Check what would be changed (dry run)
isort --check-only .
```

### Lint Code with flake8

```bash
# Lint all Python files
flake8 .

# Lint specific directory
flake8 src/

# Show statistics
flake8 --statistics .
```

### Type Check with mypy

```bash
# Type check all files
mypy .

# Type check specific file
mypy src/rag_pipeline.py

# Ignore missing imports (for optional dependencies)
mypy --ignore-missing-imports .
```

## Pre-commit Hooks

Once installed, pre-commit hooks will automatically:

1. **Format code** with Black
2. **Sort imports** with isort
3. **Lint code** with flake8
4. **Type check** with mypy (optional, can be slow)
5. **Check for** trailing whitespace, large files, YAML/JSON validity

### Bypassing Hooks (if needed)

If you need to commit without running hooks (not recommended):

```bash
git commit --no-verify -m "Emergency commit"
```

## Configuration Files

- **`.pre-commit-config.yaml`** - Pre-commit hooks configuration
- **`pyproject.toml`** - Black, isort, and mypy configuration

## IDE Integration

### VS Code

Add to `.vscode/settings.json`:

```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### PyCharm

1. Go to Settings → Tools → External Tools
2. Add Black and isort as external tools
3. Enable "Reformat code" on save

## Workflow

### Recommended Workflow

1. **Before committing:**
   ```bash
   # Format and sort automatically
   black .
   isort .
   
   # Check for issues
   flake8 .
   ```

2. **Or let pre-commit handle it:**
   ```bash
   git add .
   git commit -m "Your message"
   # Pre-commit hooks run automatically
   ```

### Continuous Integration

You can also add these checks to CI/CD:

```yaml
# Example GitHub Actions
- name: Check code quality
  run: |
    black --check .
    isort --check-only .
    flake8 .
    mypy --ignore-missing-imports .
```

## Troubleshooting

### Pre-commit not running

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install
```

### mypy errors with optional dependencies

The configuration already ignores missing imports for optional dependencies. If you see errors:

```bash
mypy --ignore-missing-imports .
```

### Black and isort conflicts

The configuration uses `isort --profile=black` to ensure compatibility. If you see conflicts:

```bash
# Run isort after black
black .
isort .
```

## Benefits

✅ **Consistent code style** across the project  
✅ **Automatic formatting** saves time  
✅ **Catch errors early** before committing  
✅ **Professional codebase** ready for review  
✅ **Team collaboration** with shared standards  

## Next Steps

1. Install the tools: `pip install black isort flake8 mypy pre-commit`
2. Set up hooks: `pre-commit install`
3. Make a test commit to see hooks in action
4. Configure your IDE for automatic formatting

For more details, see `CODE_QUALITY_REPORT.md`.

