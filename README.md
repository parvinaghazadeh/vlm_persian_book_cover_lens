# VLM PersianBookCoverLens

**Persian Book Title Recognition from Book Cover Images using Vision-Language Models**

This repository contains a Google Colab-ready Jupyter notebook for extracting Persian book titles from cover images using a Vision-Language Model (VLM). The project is designed for a course assignment on Vision-Language Models and includes dataset loading, baseline inference, evaluation metrics, LoRA/QLoRA fine-tuning code, and before/after comparison outputs.

> **Important note about runtime:** Full VLM inference and fine-tuning require a CUDA GPU runtime. The notebook includes a CPU-safe `dev_demo` mode so the project structure, dataset pipeline, metric functions, output files, and plots can be tested on a free CPU-only Colab runtime without producing misleading low-quality OCR results.

---

## Project Objective

The goal is to recognize the **Persian title of a book** from its cover image.

Given:

- Input: a book cover image
- Target output: the Persian title text

The notebook supports:

1. Loading the Persian book cover dataset.
2. Running a base VLM before fine-tuning.
3. Computing evaluation metrics.
4. Fine-tuning the VLM using LoRA/QLoRA when GPU is available.
5. Running evaluation after fine-tuning.
6. Comparing before/after results using tables, CSV files, and a metric comparison plot.

---

## Dataset

The project uses the following Hugging Face dataset:

```text
shenasa/bookroom-persian-book-covers-and-titles
```
Hugging Face page:

```text
https://huggingface.co/datasets/shenasa/bookroom-persian-book-covers-and-titles
```

The dataset contains book cover images and corresponding Persian title labels.

Default assignment-oriented split setup:

```python
TRAIN_LIMIT = 1000
EVAL_LIMIT = 200
```

For quick development tests, the notebook can use much smaller subsets.

---

## Model

The main VLM track uses:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

This model is used for image-conditioned text generation. In this project, the prompt asks the model to return only the Persian book title visible on the cover.

---

## Runtime Modes

The notebook has two clear execution modes.

### 1. `dev_demo` mode — default for CPU-only Colab

```python
RUN_MODE = "dev_demo"
```

This mode is intended for:

- free Colab CPU runtime
- testing notebook execution
- validating dataset loading
- validating metric functions
- validating CSV/plot output generation
- presenting the project pipeline when GPU is unavailable

This mode **does not claim real VLM performance**. It creates structured placeholder outputs so the notebook can run end-to-end without downloading or fine-tuning a large VLM on CPU.

### 2. `qwen_gpu` mode — real VLM experiment

```python
RUN_MODE = "qwen_gpu"
```

This mode is intended for:

- Colab GPU runtime
- Kaggle GPU notebook
- local CUDA machine
- real baseline inference
- real LoRA/QLoRA fine-tuning
- real before/after metric comparison

If `qwen_gpu` is selected and CUDA is not available, the notebook stops early with a clear error message instead of silently running a weak CPU fallback.

---

## Evaluation Metrics

The project implements the metrics requested in the assignment:

### Exact Match

Measures whether the predicted title exactly matches the ground truth.

### Normalized Exact Match

Applies Persian text normalization before exact matching. This reduces false mismatches caused by Arabic/Persian character variants, diacritics, spacing, and punctuation.

### Levenshtein Similarity

Measures character-level similarity between prediction and ground truth. This is more informative than Exact Match when the model makes small OCR mistakes.

### Word-level F1

Measures token-level overlap between predicted and true titles. This is useful when the model extracts some correct words but misses or adds others.

---

## Technologies Used

- **Python** — main programming language
- **Google Colab** — target execution environment
- **Hugging Face Datasets** — dataset loading and subset selection
- **Hugging Face Transformers** — VLM loading, processor, generation, and training interface
- **Qwen2.5-VL** — main Vision-Language Model track
- **qwen-vl-utils** — Qwen visual input preprocessing
- **PyTorch** — model execution and training backend
- **PEFT** — LoRA adapter fine-tuning
- **BitsAndBytes** — 4-bit quantization for GPU memory reduction
- **RapidFuzz** — Levenshtein similarity calculation
- **Pandas** — result tables and CSV outputs
- **Matplotlib** — metric comparison plot

---

## Main Features

- Google Colab-ready notebook
- CPU-safe development mode
- GPU-only real VLM mode
- Persian text normalization
- Exact Match metric
- Normalized Exact Match metric
- Levenshtein Similarity metric
- Word-level F1 metric
- Baseline prediction pipeline
- Fine-tuned prediction pipeline
- LoRA/QLoRA fine-tuning code
- Before/after comparison table
- CSV export of predictions
- CSV export of metric summary
- Metric comparison chart
- Clear separation between demo outputs and real model outputs
- GitHub-ready README and `.gitignore`

---

## Recommended Colab Usage

### CPU-only Colab / safe project demonstration

Use this configuration:

```python
RUN_MODE = "dev_demo"
FAST_DEV_RUN = True
```

or for a larger output table without running the VLM:

```python
RUN_MODE = "dev_demo"
FAST_DEV_RUN = False
```

This will validate the full project pipeline but will not produce real model-performance metrics.

### GPU runtime / final real experiment

Use this configuration:

```python
RUN_MODE = "qwen_gpu"
FAST_DEV_RUN = False
RUN_BASELINE_INFERENCE = True
RUN_FINETUNING = True
RUN_FINETUNED_INFERENCE = True
```

Default final limits:

```python
TRAIN_LIMIT = 1000
EVAL_LIMIT = 200
```

---

## Output Files

The notebook writes results to the `outputs/` directory:

```text
outputs/baseline_predictions.csv
outputs/finetuned_predictions.csv
outputs/metrics_summary.csv
outputs/metrics_comparison.png
outputs/sample_predictions_after_finetuning.csv
outputs/qwen_lora_adapter/
```

---

## How to Report Results

A concise report can be written as follows:

> The base VLM was evaluated on the selected test subset using Exact Match, Normalized Exact Match, Levenshtein Similarity, and Word-level F1. The model was then fine-tuned on a subset of Persian book cover images using LoRA/QLoRA. After fine-tuning, the same test subset and the same metrics were used for evaluation. The final table and plot compare model performance before and after fine-tuning.

If only CPU runtime was available:

> Due to CPU-only runtime limitations, the notebook was executed in development demonstration mode. The full GPU-based Qwen2.5-VL inference and LoRA fine-tuning implementation is included, but actual model training requires a CUDA GPU runtime.

---

## Limitations

- Real VLM fine-tuning is not practical on CPU-only Colab.
- Exact Match is strict and may score near zero even when the predicted title is partially correct.
- Persian OCR is sensitive to font, cover design, text orientation, visual clutter, and decorative typography.
- The `dev_demo` mode is only for pipeline validation and should not be reported as real model accuracy.
