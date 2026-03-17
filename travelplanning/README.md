## 🛠️ Quick Start

This domain can be run as part of the unified benchmark or independently.

### Step 1: Install Dependencies

**Note:** The unified environment is set up in the project root directory.

```bash
# Navigate to project root (if you're in travelplanning/)
cd ..

# Create a new conda environment (recommended Python 3.10)
conda create -n deepplanning python=3.10 -y

# Activate the environment
conda activate deepplanning

# Install all required packages from the unified requirements.txt
pip install -r requirements.txt

# Return to travelplanning directory
cd travelplanning
```

### Step 2: Download Data Files

**Required Files:**
- `database/database_zh.zip` - Chinese database
- `database/database_en.zip` - English database

**Download from:** [HuggingFace Dataset](https://huggingface.co/datasets/Qwen/DeepPlanning)

First, download the required data files from HuggingFace and place them in the project:

- In `travelplanning/database/`: put `database_zh.zip` and `database_en.zip`.



### Step 3: Extract Database Files

After downloading, extract the compressed travel databases:

```bash
# Navigate to the database directory
cd database

# Extract both language databases
unzip database_zh.zip    # Chinese database (flights, hotels, restaurants, attractions)
unzip database_en.zip    # English database

# Return to travelplanning directory
cd ..
```


### Step 4: Configure Model Settings

**Note:** Model configuration is shared across all domains and located in the project root.

Edit `models_config.json` in the **project root directory** (one level up from travelplanning/):

```json
{
  "models": {
    "qwen-plus": {
      "model_name": "qwen-plus",
      "model_type": "openai",
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "api_key_env": "DASHSCOPE_API_KEY"
    },
    "gpt-4o-2024-11-20": {
      "model_name": "gpt-4o-2024-11-20",
      "model_type": "openai",
      "base_url": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY"
    }
  }
}
```

**Important Note about the conversion model:**
- The conversion stage (`evaluation/convert_report.py`) uses an LLM to parse the agent-generated report into structured JSON.
- By default, `evaluation/convert_report.py` uses `conversion_model = "deepseek-reasoner"`. Ensure this model exists in `models_config.json`, or change `conversion_model` to a model you have configured (e.g., `qwen-plus` or `gpt-4o-2024-11-20`).

**Supported Model Types:**
- `openai`: OpenAI and compatible models (GPT-4, Qwen, DeepSeek, etc.)

### Step 5: Set API Keys

**Note:** API keys are configured in the project root directory.

Create a `.env` file in the **project root directory** (or `travelplanning/.env`) or set environment variables.
The runner and conversion code will attempt to load `.env` automatically.

```bash
# Option 1: Create .env file (project root or travelplanning/.env)
# Example contents:
# DASHSCOPE_API_KEY=...
# OPENAI_API_KEY=...
# DEEPSEEK_API_KEY=...

# Option 2: Set environment variables directly
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export OPENAI_API_KEY="your_openai_api_key"
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

### Step 6: Run the Benchmark

There are two entrypoints:
- `run.py`: run a single model (one language or both languages)
- `main.py`: run multiple models concurrently with a pre-check (Windows-friendly)

```bash
python run.py --model qwen-plus --language zh --workers 40 --max-llm-calls 200 --start-from inference
```

**Smart Caching & Resume Functionality:**

When using `run.py` (or `main.py`) with `--start-from inference`, the runner automatically:

1. **Checks existing results** for the same model name to avoid redundant work
2. **Scans the `reports/` folder** first to find missing report files (e.g., `id_0_report.txt`, `id_1_report.txt`, etc.)
3. **Scans the `converted_plans/` folder** to find missing converted plan files (e.g., `id_0_converted.json`, `id_1_converted.json`, etc.)
4. **Identifies missing task IDs** (out of 120 total tasks: IDs 0-119)
5. **Automatically determines the starting step:**
   - If reports are complete but converted plans are missing → starts from `conversion`
   - If reports are missing → starts from `inference`
   - If both are complete → skips the model (handled by `main.py` pre-check)

This allows you to safely interrupt and resume long-running evaluations without losing progress.

#### Option A: Run a Single Model (`run.py`)

```bash
# Run a single language
python run.py --model qwen-plus --language zh --workers 40 --max-llm-calls 200 --start-from inference

# Run both languages (omit --language)
python run.py --model qwen-plus --workers 40 --max-llm-calls 200 --start-from inference

# Rerun specific IDs only
python run.py --model qwen-plus --language zh --rerun-ids "0-10,15" --start-from inference

# Store outputs under a custom base directory (results still go into {output_dir}/{model}_{lang}/...)
python run.py --model qwen-plus --language zh --output-dir "D:\deepplanning_results" --start-from inference
```

#### Option B: Run Multiple Models Concurrently (`main.py`)

```bash
# Run multiple models for one language
python main.py --models "qwen-plus gpt-4o-2024-11-20" --language zh --workers 40 --max-llm-calls 200 --start-from inference

# Run both languages (pass empty string)
python main.py --models "qwen-plus gpt-4o-2024-11-20" --language "" --workers 40 --max-llm-calls 200 --start-from inference

# Run specific task IDs only (passed through to run.py --rerun-ids)
python main.py --models "qwen-plus" --task-id "0-10,15" --language zh --start-from inference
```

## 🔄 Understanding the Pipeline

The benchmark runs in three stages:

#### Stage 1: Inference (Agent Planning)
**What it does:** 
- Loads travel planning tasks from `data/travelplanning_query_{lang}.json`
- Calls the LLM agent to generate travel plans
- Agent uses tools to query database (flights, hotels, restaurants, attractions)
- Saves agent trajectories and execution logs
- Generates human-readable reports following the required format

**Output:**
```
results/{model}_{lang}/
├── trajectories/     # Agent execution traces
│   └── id_0_trajectory.json
└── reports/          # Human-readable reports
    └── id_0_report.txt
```

#### Stage 2: Conversion (Plan Parsing)
**What it does:**
- Uses an LLM (default config name: `deepseek-reasoner`, configurable in `evaluation/convert_report.py`) to convert plans
- **Parses Markdown-formatted travel plans** from agent output
- **Converts to standardized JSON format** for automated evaluation
- Stores converted plans in `converted_plans/` directory
- Validates plan structure and completeness

**Why conversion is needed:** The agent generates human-readable plans in Markdown format, but the evaluation code requires structured JSON data to automatically score compliance with constraints and calculate metrics.

**Output:**
```
results/{model}_{lang}/
├── converted_plans/  # Structured travel plans
│   └── id_0_converted.json
```


#### Stage 3: Evaluation
**What it does:**
- Checks delivery rate (was a plan generated?)
- Evaluates commonsense score (8 dimensions)
- Validates personalized constraints
- Calculates final scores

**Output:**
```
results/{model}_{lang}/
└── evaluation/
    ├── evaluation_summary.json      # Overall metrics and statistics
    ├── id_0_score.json              # Individual task scores
    ├── id_1_score.json
    └── ...                           # One score file per task
```

## 📊 Viewing Results

#### Overall Statistics

```bash
cat results/{model}_{lang}/evaluation/evaluation_summary.json
```

**Example Output:**
```json
{
  "total_test_samples": 120,
  "evaluation_success_count": 115,
  "metrics": {
    "delivery_rate": 0.958,
    "commonsense_score": 0.875,
    "personalized_score": 0.742,
    "composite_score": 0.809,
    "case_acc": 0.683
  }
}
```

#### Per-Task Details

```bash
# View detailed score for a specific task
cat results/{model}_{lang}/evaluation/id_0_score.json


# View human-readable report for a specific task
cat results/{model}_{lang}/reports/id_0_report.txt
```

#### Error Analysis

The summary includes error statistics showing common failure patterns:

```json
"error_statistics": [
  {
    "rank": 1,
    "error_type": "[Hard] train_seat_status",
    "count": 15,
    "affected_samples": ["0", "12", "25", ...]
  }
]
```
