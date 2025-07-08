# Data Leakage v2

**Data Leakage v2** is an **end-to-end toolkit** for **simulating, detecting, and quantifying training-data leakage** in code-generation LLMs.
It contains:

* dataset constructors that splice *original*, *variant* and *multi-source* leaks into benchmarks such as **HumanEval** and **GSM-8K**
* LoRA fine-tuning scripts that deliberately **inject contaminated examples** so you can study memorisation dynamics ([raw.githubusercontent.com][2])
* two lightweight detectors:

  * **CDD** – *Contamination Detection by (normalized) Levenshtein Distance* ([raw.githubusercontent.com][3])
  * **TED** – *Token-Edit-Distance drop-in that re-computes pass\@k after excluding near-copies* ([raw.githubusercontent.com][4])
* baseline scorers (Min-k Prob, F1-emb, etc.) for fair comparison ([raw.githubusercontent.com][5])

```text
Data_Leakage_v2
│
├── dataset/                         # original contamination-detection splits
├── dataset_gsm/                     # GSM-8K-style splits
├── baselines/                       # baselines for Min-k Prob, F1-emb, etc.
├── eval/
│   └── baselines.py                 # reproducible baseline launcher
├── CDD.py                           # Contamination Detection via Edit Distance
├── CDD_Detect.py                    # CDD variant for pair-wise model analysis
├── TED.py                           # Token-Edit-Distance pass@k evaluator
├── leakage_v2.py / leakage_v3.py    # LoRA fine-tuning to *inject* leakage
├── construct_*_dataset_*.py         # scripts that build detection datasets
├── finetune_lora.py                 # generic PEFT trainer wrapper
├── *.sh                             # turnkey replication scripts
└── README.md                        # minimal Chinese stub (overwritten here)
```

The code was used in our 2024 TPAMI / ACL study on **model robustness under code-data leakage** (PKU SEKE Lab, with Dr. Dong Yihong).

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/XMZhangAI/Data_Leakage_v2.git
cd Data_Leakage_v2
python -m venv .venv && source .venv/bin/activate

# 2. Install deps  (CUDA-enabled PyTorch assumed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets evaluate peft tiktoken scikit-learn numpy

# 3. Build a contamination-detection dataset (HumanEval original)
python construct_original_data_contamination_detection_dataset_truncate_iteration.py \
       --input_humaneval_json humaneval.jsonl \
       --output_dir datasets/original_detection_epoch2

# 4. Inject leakage via LoRA fine-tuning
python leakage_v2.py \
       --dataset_path datasets/original_detection_epoch2 \
       --model_name Llama-2-7b-hf \
       --output_dir checkpoints/llama2_leakage

# 5. Detect with CDD
python CDD.py \
       --alpha 0.05  \
       --xi 0.01     \
       --input_path datasets/detect_dataset_all_v3/original_data_contamination_detection_dataset_truncate_epoch2 \
       --model CodeLlama-7b

# 6. Evaluate pass@k with TED (exclude top-τ similar completions)
python TED.py \
       --tau 2 \
       --Occurrences 20 \
       --input_path Sample_Results/CodeLlama-7B
```

> **GPU memory**: LoRA scripts default to rank = 128 and full-precision; adjust `--lora_rank` / `--bf16` for smaller cards.

---

## Detectors At a Glance

| Detector | Signal                                                                               | Strengths                                                                | When to use                                    |
| -------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ | ---------------------------------------------- |
| **CDD**  | Normalised edit-distance between *greedy* sample and *T* temperature-diverse samples | Fast (tokeniser-level), no gradients                                     | Quick leak triage for any autoregressive model |
| **TED**  | Impact on pass\@k after removing completions within τ tokens of the greedy baseline  | Correlates with *semantic* memorisation; integrates with exec-based eval | Research on *down-stream* metric inflation     |

Implementation details live in `CDD.py` and `TED.py`; both expose CLIs with `--alpha` (distance threshold) and `--tau` (edit-distance cut-off) respectively.

---

## Baselines

```text
eval/
└── baselines.py     # reproduces Min-k Prob, F1-emb, LatestEval, RephraseEval
```

Each baseline yields **Precision / Recall / F1 / AUC** on the chosen detection split.
To add a new heuristic, drop a function that maps token-probability blobs (`ground_truth_prob`) or embeddings to a scalar score, then register it in the main loop.

---

## Results Reproduction

Shell scripts in the repo reproduce all major tables:

* `run_codellama_new_humaneval_0217.sh` – CodeLlama-7B w/ HumanEval
* `run_bloom_new_gsm_0217.sh`          – BLOOM-176B w/ GSM-8K
* `llama_gsm8k.sh` / `Mistral_gsm8k.sh` – mixed GSM-8K contamination study

They follow the pattern **construct → fine-tune → generate → detect → evaluate**.

---

## Citation

```bibtex
@misc{donggeneralization,
  title   = {Generalization or Memorization: Evaluating Data Contamination for Large Language Models},
  author  = {Dong, Yihong and Jiang, Xue and Zhang, Xuanming and Liu, Huanyu and Jin, Zhi and Gu, Bin and Yang, Mengfei and Li, Ge},
  journal = {Peking University Computer Science},
  year    = {2025},
  month   = {Jan},
  url     = {https://www.researchgate.net/publication/387596869_Generalization_or_Memorization_Evaluating_Data_Contamination_for_Large_Language_Models},
}
```

---

## License

Released under the **MIT License**.  See `LICENSE` for details.

> **Research-only notice**: The included data splits are derived from *public* benchmarks but deliberately leak subsets of the test answers. **Do not** fine-tune production models with these files.

---

## Acknowledgements

This project was developed at the **SEKE Lab, Peking University** with support from Dr. **Dong Yihong**.
Thanks to the open-source **Hugging Face Transformers / Datasets / PEFT** communities for the excellent tooling.

*Happy hacking & safe benchmarking!*

[1]: https://github.com/XMZhangAI/Data_Leakage_v2 "GitHub - XMZhangAI/Data_Leakage_v2: [2024TPAMI][2024ACL]PKU SEKE Lab, with Dr. Dong Yihong"
[2]: https://raw.githubusercontent.com/XMZhangAI/Data_Leakage_v2/main/leakage_v2.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/XMZhangAI/Data_Leakage_v2/main/CDD.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/XMZhangAI/Data_Leakage_v2/main/TED.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/XMZhangAI/Data_Leakage_v2/main/eval/baselines.py "raw.githubusercontent.com"
