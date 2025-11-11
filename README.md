# SimpleQuestions-3000: Deepseek API Evaluation

This repository evaluates Deepseek API on the SimpleQuestions-3000 question-answering dataset.

## 📊 Experimental Results

**Deepseek API Performance on test.json (n=1000):**
- **Exact Match (EM):** 19.9%
- **F1 Score:** 28.87%
- **Model:** deepseek-chat
- **Strategy:** Zero-shot (no few-shot examples)
- **Temperature:** 0.0

## 📁 Dataset

SimpleQuestions-3000 is a factual question-answering dataset with:
- `train.json`: 18,000 samples
- `dev.json`: 1,200 samples  
- `test.json`: 6,000 samples (evaluated first 1,000 for this experiment)

Each sample contains:
```json
{
  "question": "Where did Sholom Schwartzbard die",
  "answers": ["Cape Town"]
}
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Deepseek API key
- Install dependencies:

```bat
pip install requests tqdm numpy
```

### Run Evaluation

**1. Set Deepseek API key**

```bat
set DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

**2. Run Deepseek API on test set**

```bat
python scripts\deepseek_api_runner.py --input test.json --output results\deepseek_api_test.json --model deepseek-chat --temperature 0.0
```

**3. Evaluate results**

```bat
python scripts\evaluator.py --mode chatgpt --preds results\deepseek_api_test.json --out results\deepseek_report.json
type results\deepseek_report.json
```

## 📂 Project Structure

```
SimpleQuestions-3000/
├── train.json                         # Training data (18K samples)
├── dev.json                           # Development data (1.2K samples)
├── test.json                          # Test data (6K samples)
├── scripts/
│   ├── deepseek_api_runner.py        # Deepseek API caller
│   ├── evaluator.py                   # Metrics computation (EM/F1)
│   └── utils.py                       # Normalization & helper functions
├── results/
│   ├── deepseek_api_test.json        # Raw predictions
│   └── deepseek_report.json          # Evaluation metrics
└── README.md
```

## 📈 Output Files

- **`results/deepseek_api_test.json`**  
  Raw Deepseek API predictions for each question:
  ```json
  {
    "id": 0,
    "question": "Where did Sholom Schwartzbard die",
    "pred": "Cape Town",
    "answers": ["Cape Town"]
  }
  ```

- **`results/deepseek_report.json`**  
  Aggregate metrics:
  ```json
  {
    "EM": 0.199,
    "F1": 0.2887065825672014,
    "n": 1000
  }
  ```

## 🔧 Configuration

### Model Parameters

- **Model:** `deepseek-chat` (default)
- **Temperature:** 0.0 (deterministic)
- **Max Tokens:** 64 (short answer generation)
- **Retry:** 3 attempts with exponential backoff

### Evaluation Metrics

- **Exact Match (EM):** Percentage of predictions that exactly match ground truth after normalization (lowercase, punctuation removal, whitespace trimming)
- **F1 Score:** Token-level overlap between prediction and ground truth (precision × recall harmonic mean)

## 💡 Notes

- **API Format:** Uses OpenAI-compatible chat completion endpoint
- **Prompt Strategy:** Zero-shot with system message: *"You are a concise factual question-answering assistant. Answer as briefly as possible with just the answer, no explanation."*
- **Normalization:** Both predictions and ground truth are normalized before comparison (case-insensitive, punctuation removed)
- **Cost:** ~1000 API calls for full test set evaluation

## 🎯 Potential Improvements

1. **Few-shot prompting:** Add 3-8 examples from `dev.json` → Expected +5-10% EM
2. **Prompt engineering:** More specific instructions for entity/date extraction
3. **Answer post-processing:** Extract first noun phrase or entity from longer responses
4. **Question-type stratification:** Different prompts for who/where/what/when questions

## 📝 Citation

Dataset: SimpleQuestions (Facebook AI Research)  
Original paper: *"Large-scale Simple Question Answering with Memory Networks"* (Bordes et al., 2015)

---

**Experiment completed:** November 11, 2025  
**Evaluated by:** Deepseek API (deepseek-chat)
