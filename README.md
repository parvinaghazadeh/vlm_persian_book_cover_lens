# PersianBookCoverLens: Persian Book Cover Title Recognition

**PersianBookCoverLens** is a reproducible Persian book-cover title recognition project for a Vision-Language Models (VLM) course assignment. The task is to extract the Persian title of a book from its cover image, evaluate the model before fine-tuning, fine-tune it on a subset of the training data, and compare performance after fine-tuning.

The project is designed for two execution realities:

1. **CPU-compatible final run** for free Google Colab sessions without GPU access.
2. **Optional Qwen2.5-VL track** for GPU runtimes.

The default notebook configuration uses the CPU-compatible track so that the project can be tested and submitted without crashing in a CPU-only Colab environment.

---

## Assignment Mapping

| Assignment requirement | Implemented in this project |
|---|---|
| Load Persian book-cover dataset | Yes |
| Select train subset | Yes |
| Use test subset for evaluation | Yes |
| Extract Persian title from cover image | Yes |
| Compute Exact Match | Yes |
| Fine-tune on training subset | Yes in CPU OCR track; optional Qwen/VLM track for GPU |
| Re-evaluate after fine-tuning | Yes |
| Compare before/after metrics | Yes |
| Plot metric improvement | Yes |
| Optional Levenshtein Accuracy/Similarity | Yes |
| Optional Word-level F1 | Yes |

---

## Dataset

The project uses the Hugging Face dataset:

```python
DATASET_ID = "shenasa/bookroom-persian-book-covers-and-titles"
```

The dataset contains Persian book-cover images and title text labels. The notebook expects the following columns:

```text
image
text
```

---

## Model Tracks

### 1. CPU-compatible OCR fine-tuning track

This is the default and recommended track when running on free Colab without GPU.

It trains a small image-to-character OCR model:

- compact CNN image encoder;
- GRU character decoder;
- Persian text normalization;
- greedy decoding;
- before/after fine-tuning evaluation.

This track is intentionally lightweight. It is not meant to outperform large VLMs, but it provides a real, reproducible fine-tuning experiment under CPU limitations.

Default CPU settings:

```python
EXPERIMENT_TRACK = "cpu_light_ocr"
FAST_DEV_RUN = False
CPU_SAFE_MODE = True
TRAIN_LIMIT = 200
EVAL_LIMIT = 50
EPOCHS = 3
```

For a quick smoke test:

```python
FAST_DEV_RUN = True
```

### 2. Optional Qwen2.5-VL track

The notebook also includes an optional Qwen2.5-VL inference section:

```python
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
```

Use this track only when a GPU runtime is available:

```python
EXPERIMENT_TRACK = "qwen_vlm"
CPU_SAFE_MODE = False
```

Full Qwen/VLM fine-tuning is not practical on CPU-only Colab. The notebook keeps this track separate to avoid runtime crashes.

---

## Metrics

The notebook computes the following metrics:

### 1. Raw Exact Match

Checks whether the predicted title exactly matches the ground-truth title without normalization.

### 2. Normalized Exact Match

Applies Persian text normalization before comparison. This handles common differences such as Arabic/Persian Yeh and Kaf, half-spaces, punctuation, and digit forms.

### 3. Levenshtein Similarity

Measures character-level similarity between prediction and ground truth:

```text
1 - edit_distance / max_length
```

This is useful for OCR because small spelling or character mistakes should not be treated the same as completely wrong predictions.

### 4. Word-level F1

Measures word overlap between prediction and ground truth. This is useful when part of the title is recognized correctly.

---

## Output Files

After a successful run, the notebook creates:

```text
outputs/baseline_predictions.csv
outputs/finetuned_predictions.csv
outputs/metrics_summary.csv
outputs/metrics_comparison.png
outputs/training_history.csv
outputs/tiny_cover_ocr_finetuned.pt
```

Recommended files to include in your report or presentation:

- `metrics_summary.csv`
- `metrics_comparison.png`
- sample rows from `baseline_predictions.csv`
- sample rows from `finetuned_predictions.csv`

---

## How to Run in Free Google Colab Without GPU

1. Open the notebook in Google Colab.
2. Keep the runtime as the default CPU runtime.
3. Run the installation cell.
4. If Colab requests a restart, restart the runtime once.
5. Continue from the imports/configuration section.
6. Keep the default configuration:

```python
EXPERIMENT_TRACK = "cpu_light_ocr"
FAST_DEV_RUN = False
CPU_SAFE_MODE = True
```

If the run is slow, reduce:

```python
TRAIN_LIMIT = 100
EVAL_LIMIT = 30
EPOCHS = 2
```

---

## How to Run with GPU

If you have access to a GPU runtime, you can use the Qwen/VLM track:

```python
EXPERIMENT_TRACK = "qwen_vlm"
CPU_SAFE_MODE = False
```

Also set:

```python
INSTALL_QWEN_DEPS = True
```

in the installation cell.

For assignment-scale evaluation, use:

```python
TRAIN_LIMIT = 1000
EVAL_LIMIT = 200
```

---

## Important Note About CPU-only Fine-tuning

The original assignment asks for VLM fine-tuning. Full VLM fine-tuning with Qwen2.5-VL is not realistic on CPU-only Colab. Therefore, this project includes a CPU-compatible OCR fine-tuning track to demonstrate the same experimental structure:

1. evaluate before fine-tuning;
2. fine-tune on Persian book-cover data;
3. evaluate after fine-tuning;
4. compare Exact Match, Levenshtein Similarity, and Word-level F1.

This makes the notebook executable and reproducible under the available hardware while keeping the Qwen/VLM path available for GPU-based extension.

Suggested explanation in your presentation:

> The original VLM fine-tuning path is implemented for GPU runtimes. Because my available Colab runtime was CPU-only, I added a CPU-compatible OCR fine-tuning track to demonstrate the required before/after fine-tuning comparison in a reproducible way. The same evaluation metrics are used in both tracks.

---

## Repository Structure

```text
vlm_persian_book_cover_lens/
├── PersianBookCoverLens.ipynb
├── README.md
├── requirements-colab.txt
├── .gitignore
└── outputs/
    └── .gitkeep
```

---

## Troubleshooting

### PIL / Pillow import error

If you see a Pillow/PIL error, set this in the installation cell:

```python
REPAIR_PIL = True
```

Run the install cell, restart runtime, then continue.

### CPU run is too slow

Reduce the CPU limits:

```python
TRAIN_LIMIT = 100
EVAL_LIMIT = 30
EPOCHS = 2
```

### Qwen track fails on CPU

That is expected. Use the Qwen track only with GPU.

---

## Limitations

- The CPU-compatible model is intentionally small and may not achieve high accuracy on complex book covers.
- Book covers often contain multiple text regions, author names, subtitles, decorative fonts, and low-resolution text.
- Exact Match is very strict for OCR tasks; Levenshtein Similarity and Word-level F1 are more informative.
- Full VLM fine-tuning requires GPU resources.

---

## Future Work

- Fine-tune Qwen2.5-VL or a smaller modern VLM with LoRA/QLoRA on a GPU runtime.
- Add a text-region detector before OCR.
- Use stronger Persian OCR models as initialization.
- Use synthetic Persian title rendering for OCR pretraining.
- Add data augmentation such as small rotations, brightness changes, and crops.
- Add evaluation on the full 200-sample test subset when GPU or longer CPU runtime is available.

---

## License

This project is prepared for educational use. Please check the licenses of the dataset and any external model before publishing trained weights.
