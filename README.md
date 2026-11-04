# Surpassing Dictionary-Based G2P for Japanese with a 1B Language Model

Code, evaluation data, and notebooks accompanying the paper:

> **Surpassing Dictionary-Based G2P for Japanese with a 1B Language Model:
> Human-Annotated Training Data and Deterministic Decoding**
> Takafumi Fujii, 2026. (arXiv link TBA)

A 1B-parameter SLM (sarashina2.2-1b-instruct + LoRA) fine-tuned **only on
human-annotated resources and rule-based generation** (no LLM-generated data),
combined with a deterministic inference configuration (sentence splitting,
kana-constrained decoding, beam search), outperforms the de-facto standard
dictionary-based G2P (**pyopenjtalk**) on **all seven evaluation items**, with
statistically significant wins on four:

| Evaluation item | pyopenjtalk | Proposed | Paired test |
|---|---|---|---|
| jvs overall CER ↓ | 0.933% | **0.924%** | fix 92 / regress 91 (parity) |
| jvs digit-slice CER ↓ | 2.36% | **2.15%** | (diagnostic) |
| Joyo ↑ | 96.54% | **96.96%** | p=2.5e-3 |
| Joyo Parakeet ed. ↑ | 97.01% | **97.68%** | p=2.8e-7 |
| KWDLC-test CER ↓ | 1.236% | **0.934%** | p=1.8e-6 |
| Heteronym set ↑ | 59.2% | **69.1%** | p=3.1e-5 |
| g2pbench ↑ | 65.00% | **70.63%** | p=0.14 (n.s.) |

The system never consults the dictionary at inference time.

## Contents

```
notebooks/
  colab_slm_g2p_v11_full.ipynb     # end-to-end pipeline in one notebook:
                                   # dataset construction (components (a)-(h),
                                   # contamination audit) -> LoRA training (~30h,
                                   # auto-resume) -> final-system inference for
                                   # all benchmarks (sentence split + kana mask
                                   # + beam-4)
  inference_demo.ipynb             # try the released model on your own text
                                   # (loads the adapter from Hugging Face)
eval/
  eval_jvs.py                      # jvs scorer (CER, paired 2x2, digit slice)
  eval_hard_set.py                 # heteronym-set scorer (alignment-based span
                                   # judgment, pronunciation-equivalence norm)
  eval_aligned_bench.py            # Joyo / Parakeet / g2pbench / KWDLC scorers
data/
  heteronym_eval_set_v1.csv        # 300 sentences, blind-annotated gold readings
  HETERONYM_SET.md                 # construction procedure
hf_model/
  README.md                        # model card for the LoRA adapter (HF upload)
  adapter_model.safetensors etc.   # the trained v11 adapter (54MB)
```

**Model**: the trained LoRA adapter is released on Hugging Face:
[t2-fujii/slm-ja-g2p-v11-lora](https://huggingface.co/t2-fujii/slm-ja-g2p-v11-lora).
`notebooks/inference_demo.ipynb` shows end-user inference with the paper's
exact configuration.

## Reproducing the results

1. **Run the pipeline**: upload `data/heteronym_eval_set_v1.csv` to the
   working directory, then execute `notebooks/colab_slm_g2p_v11_full.ipynb` on
   Colab (A100). It builds all data components from pinned public sources with
   fixed seeds, runs a SHA-1 audit verifying zero overlap between training rows
   and every evaluation sentence, trains the LoRA adapter, and generates
   predictions for all benchmarks with the deterministic inference
   configuration. (With the [released adapter](https://huggingface.co/t2-fujii/slm-ja-g2p-v11-lora),
   you can skip training: run the construction cells, place the adapter at
   `results_v11/final`, and run the evaluation cells.)
2. **Score** locally:
   ```bash
   # dependencies: the official jvs evaluation data and the g2pbench items
   git clone https://github.com/CyberAgentAILab/jvs_nonpara_kana external/jvs_nonpara_kana
   git clone https://github.com/filmapp/ja-tts-g2p-bench external/ja_tts_g2p_bench
   pip install pyopenjtalk sudachipy sudachidict_core jiwer datasets

   python eval/eval_jvs.py external/jvs_nonpara_kana/jvs_nonpara_kana.csv <jvs3000_preds.tsv>
   python eval/eval_hard_set.py data/heteronym_eval_set_v1.csv <hardset_preds.tsv>
   python eval/eval_aligned_bench.py joyo     <joyo_preds.tsv>
   python eval/eval_aligned_bench.py parakeet <parakeet_preds.tsv>
   python eval/eval_aligned_bench.py g2pbench <g2pbench_preds.tsv>
   python eval/eval_aligned_bench.py kwdlc    <kwdlc_test_gold.json> <kwdlc_test_preds.tsv>
   ```
   Both systems (model and pyopenjtalk) are always scored by the same judge
   with the same pronunciation-equivalence normalization.

   Note: `kwdlc_test_gold.json` is written by the notebook's data-construction
   phase (pinned KWDLC commit, deterministic), before training starts.

## The heteronym evaluation set

300 real Wikipedia sentences targeting **common words whose reading depends on
context** (e.g., 行く/行う, 通る/通う, 研究所 -jo/-sho, 世論, 重複). Selection
is fully mechanical (top-300 pyopenjtalk×Sudachi disagreement words, one
sentence each, fixed seed); gold readings were annotated blind by the author
before any model was evaluated. pyopenjtalk scores 59.2% on this set.
See `data/HETERONYM_SET.md` for the exact procedure. Also mirrored on Hugging
Face: [t2-fujii/ja-heteronym-reading-eval](https://huggingface.co/datasets/t2-fujii/ja-heteronym-reading-eval).

## Licenses

- Code (`eval/`, `notebooks/`): MIT.
- `data/heteronym_eval_set_v1.csv`: sentences are from Japanese Wikipedia and
  are distributed under **CC BY-SA 4.0**; the gold reading annotations are
  released under the same license.
- Training-data sources retain their own licenses (Wikipedia CC BY-SA,
  Aozora Bunko, KWDLC, ROHAN4600 CC0, SudachiDict Apache-2.0,
  jsut-label CC BY-SA 4.0).

## Citation

```bibtex
@misc{fujii2026slmg2p,
  title  = {Surpassing Dictionary-Based G2P for Japanese with a 1B Language
            Model: Human-Annotated Training Data and Deterministic Decoding},
  author = {Fujii, Takafumi},
  year   = {2026},
  note   = {arXiv preprint (ID TBA)}
}
```
