# VLM PersianBookCoverLens

**Persian Book Title Recognition from Book Cover Images using Vision-Language Models**

This repository implements a course-level Vision-Language Model (VLM) project for extracting **Persian book titles** from book-cover images and improving recognition accuracy through **LoRA/QLoRA fine-tuning**.

The notebook is designed to run in **Google Colab** and to produce clean evaluation outputs.

---

## Project Goal

Given an image of a Persian book cover, the model should return only the **main Persian title**.

The model is instructed not to include:

- author names
- translator names
- publisher names
- prices
- edition numbers
- explanations or extra descriptions

---

## Dataset

Dataset: [`shenasa/bookroom-persian-book-covers-and-titles`](https://huggingface.co/datasets/shenasa/bookroom-persian-book-covers-and-titles)

The dataset contains:

- `image`: book-cover image
- `text`: Persian book title

For the final experiment:

- Train subset: 1,000 samples from the `train` split
- Test subset: first 200 samples from the `test` split

The notebook also includes a `FAST_DEV_RUN=True` mode for quick testing with a much smaller subset.

---

## Model

Base model:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

This model was selected because it is a relatively small open VLM from the Qwen2.5-VL family and is suitable for OCR-like image-to-text extraction tasks.

---

## Method

The project follows these steps:

1. Load the Hugging Face dataset.
2. Select train and test subsets.
3. Run the base VLM on the test subset.
4. Compute baseline metrics.
5. Fine-tune the model using supervised LoRA/QLoRA training.
6. Evaluate the fine-tuned model on the same test subset.
7. Compare before/after results using tables and charts.

---

## Metrics

The notebook reports three metrics:

| Metric | Description |
|---|---|
| Exact Match | Percentage of predictions that exactly match the normalized reference title |
| Levenshtein Similarity | Character-level similarity between prediction and reference |
| Word-level F1 | Word overlap between prediction and reference |

Exact Match is strict, so Levenshtein Similarity and Word-level F1 are also reported to capture partial improvements.

---

## Repository Structure

```text
.
├── outputs/
├── PersianBookCoverLens.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

After running the notebook, outputs are saved in Colab under:

```text
/content/persian_book_title_vlm_outputs
```

Expected outputs:

```text
baseline_predictions.csv
finetuned_predictions.csv
metric_comparison.csv
metric_comparison.png
error_analysis.csv
qwen25vl_persian_title_lora/
```

---

## How to Run in Google Colab

1. Upload or open `Persian_Book_Title_VLM_Qwen25VL.ipynb` in Google Colab.
2. Go to:

```text
Runtime → Change runtime type → GPU
```

3. First run with:

```python
FAST_DEV_RUN = True
```

4. After the smoke test succeeds, run the final version with:

```python
FAST_DEV_RUN = False
TRAIN_SUBSET_SIZE = 1000
TEST_SUBSET_SIZE = 200
```

5. Run all notebook cells in order.
6. Download the output CSV files and chart from:

```text
/content/persian_book_title_vlm_outputs
```

---

## Important Runtime Notes

Real VLM inference and fine-tuning require a CUDA GPU. If the notebook is run without a GPU, the dataset and metric cells still execute, but VLM inference and fine-tuning are skipped to avoid fake results and memory crashes.

For Colab Free, training may still be slow. If you receive an out-of-memory error:

- keep `FAST_DEV_RUN=True` for testing
- reduce `MAX_PIXELS`
- reduce `TRAIN_SUBSET_SIZE`
- keep batch size at 1
- use gradient accumulation instead of increasing batch size

---

## Final Report Template

After running the final experiment, report:

```text
Base VLM:
- Exact Match: ...
- Levenshtein Similarity: ...
- Word-level F1: ...

Fine-tuned VLM:
- Exact Match: ...
- Levenshtein Similarity: ...
- Word-level F1: ...
```

Then include `metric_comparison.png` in your GitHub README or final presentation.

---

## Step-by-Step Testing Guide

Use the notebook in this order:

### Test 1 — Environment Check

Run the first GPU check cell.

Expected output:

```text
CUDA available: True
GPU: Tesla T4 / L4 / A100 / similar
```

If CUDA is not available, switch Colab to GPU runtime.

### Test 2 — Dataset Loading

Run the dataset loading cell.

Expected result:

- the dataset loads without errors
- `train` and `test` splits are printed
- columns include `image` and `text`

### Test 3 — Sample Visualization

Run the sample visualization cell.

Expected result:

- a few book covers are displayed
- each image has a Persian title above it

### Test 4 — Metric Functions

Run the metric unit-test cell.

Expected result:

- a Python dictionary with `exact_match`, `levenshtein_similarity`, `word_f1`, and `n_samples`

### Test 5 — Baseline VLM Evaluation

Run the model loading and baseline inference cells.

Expected result:

- base model predictions are generated
- `baseline_predictions.csv` is saved
- baseline metric values are printed

### Test 6 — Fine-Tuning Smoke Test

Keep:

```python
FAST_DEV_RUN = True
```

Expected result:

- training runs only for a few steps
- a LoRA adapter folder is saved
- no out-of-memory error occurs

### Test 7 — Final Training Run

After the smoke test succeeds, set:

```python
FAST_DEV_RUN = False
TRAIN_SUBSET_SIZE = 1000
TEST_SUBSET_SIZE = 200
```

Expected result:

- training runs on 1,000 training samples
- evaluation runs on 200 test samples
- final CSV files and chart are saved

### Test 8 — Final Comparison

Run the comparison section.

Expected result:

- `metric_comparison.csv`
- `metric_comparison.png`
- before/after metric table in the notebook

---

## Technologies

- Python
- Google Colab
- Hugging Face Datasets
- Hugging Face Transformers
- Qwen2.5-VL
- PEFT
- LoRA / QLoRA
- bitsandbytes
- Pandas
- Matplotlib
- RapidFuzz

---

## Acknowledgements

- Dataset: `shenasa/bookroom-persian-book-covers-and-titles`
- Base VLM: `Qwen/Qwen2.5-VL-3B-Instruct`
- Fine-tuning approach: parameter-efficient supervised fine-tuning with LoRA/QLoRA

---

## License

This repository is intended for educational use.