## 🛠️ Quick Start

This domain can be run independently from this folder. The entrypoint is `main.py` (inference + evaluation + statistics).

### Step 1: Install Dependencies

Dependencies are shared across all domains and installed from the project root:

```bash
cd ..
pip install -r requirements.txt
cd shoppingplanning
```

### Step 2: Check Data Layout

This folder expects:

```
shoppingplanning/
├── data/
│   ├── level_1_query_meta.json
│   ├── level_2_query_meta.json
│   └── level_3_query_meta.json
└── database/
    ├── database_level1/
    │   ├── case_0/
    │   └── ...
    ├── database_level2/
    └── database_level3/
```

If `shoppingplanning/database/` is missing, download the ShoppingPlanning databases from the [HuggingFace Dataset](https://huggingface.co/datasets/Qwen/DeepPlanning) and extract them into `shoppingplanning/database/` as `database_level1/`, `database_level2/`, `database_level3/`.

### Step 3: Configure Models

Model configs are loaded from `models_config.json` in the project root (preferred) or `shoppingplanning/models_config.json` (optional).

Example snippet (project root `models_config.json`):

```json
{
  "models": {
    "minimax-m2.5": {
      "model_name": "minimax-m2.5",
      "model_type": "openai",
      "base_url": "https://api.minimaxi.com/v1",
      "api_key_env": "MINIMAX_API_KEY",
      "temperature": 1.0
    }
  }
}
```

### Step 4: Set API Keys

The runner reads API keys from environment variables (based on each model’s `api_key_env`). It can also load a `.env` file from:
- Project root (`../.env`, preferred if present)
- Domain root (`./.env`)

PowerShell example:

```powershell
$env:OPENAI_API_KEY="your_key"
```

### Step 5: Run the Benchmark (Recommended)

Run inference + evaluation + cross-level statistics:

```bash
python main.py --models "minimax-m2.5" --levels "1 2 3" --workers 50 --max-llm-calls 200
```

Useful options:
- `--models`: space-separated model config names (must exist in `models_config.json`)
- `--levels`: space-separated levels (`1`, `2`, `3`)
- `--workers`: parallel workers for inference
- `--max-llm-calls`: maximum LLM calls per sample
- `--rerun-ids`: only run certain sample IDs (`"3,17,42"` or `"0-10,15"`)

### Optional: Run Only Inference (Single Level)

`run.py` runs inference for a single level. The `--database-dir` must point to a directory that directly contains `case_{id}/` folders.

```bash
python run.py --model minimax-m2.5 --level 1 --workers 50 --max-llm-calls 400 --database-dir database/database_level1
```

## 🔄 Understanding the Pipeline

### Stage 1: Inference

- Loads tasks from `data/level_{level}_query_meta.json`
- Runs the tool-using agent against a per-run isolated database directory
- Writes agent traces into each case directory (`messages.json`) and updates `cart.json`

Case directory contents:

```
case_0/
├── messages.json          # Agent execution traces
├── cart.json              # Final shopping cart
├── validation_cases.json  # Ground truth
├── products.jsonl
└── user_info.json
```

### Stage 2: Evaluation

`main.py` moves the isolated database into:

```
database_infered/{MODEL}/{BATCH_TIMESTAMP}/level{LEVEL}/
```

Then evaluation writes reports to:

```
result_report/{MODEL}/{BATCH_TIMESTAMP}/level{LEVEL}/
├── summary_report.json
├── case_0_report.json
└── ...
```

Cross-level statistics are saved to:

```
result_report/{MODEL}/{BATCH_TIMESTAMP}/statistics.json
```

## 📊 Viewing Results

Level summary:

```bash
cat result_report/{MODEL}/{BATCH_TIMESTAMP}/level{LEVEL}/summary_report.json
```

Cross-level statistics:

```bash
cat result_report/{MODEL}/{BATCH_TIMESTAMP}/statistics.json
```

Per-case report:

```bash
cat result_report/{MODEL}/{BATCH_TIMESTAMP}/level{LEVEL}/case_0_report.json
```

## 📝 Notes

- `main.py` creates an isolated database per run and is safe for concurrent runs.
- Reports are always saved even if a model has a high incomplete rate (see `valid` in reports). 

