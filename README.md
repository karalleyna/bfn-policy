# Bayesian Flow Network(BFN) Policy

## 📦 Setup

### 1. Create and activate the Conda environment

```bash
conda create -n bfn-pol python=3.11
conda activate bfn-pol
```

### 2. Install development dependencies

```bash
pip install -r requirements-dev.txt
```

### 3. Enable pre-commit hooks (recommended)

```bash
pre-commit install
```

---

## 🧪 Development Workflow

### Format and lint your code

```bash
./scripts/format_all.sh
```

Or manually:

```bash
black .
isort .
flake8 .
```

### Run pre-commit checks

```bash
pre-commit run --all-files
```
