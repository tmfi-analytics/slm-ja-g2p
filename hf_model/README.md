---
license: mit
base_model: sbintuitions/sarashina2.2-1b-instruct-v0.1
language:
- ja
library_name: peft
pipeline_tag: text-generation
tags:
- japanese
- g2p
- grapheme-to-phoneme
- text-to-speech
- reading-estimation
- lora
---

# SLM-JA-G2P v11 — Japanese reading estimation (yomi) with a 1B SLM

LoRA adapter for **sbintuitions/sarashina2.2-1b-instruct-v0.1** that turns the
model into a Japanese grapheme-to-phoneme (G2P) / reading estimator: input a
sentence, output its pronunciation as katakana (particles in pronounced form,
e.g. は→ワ).

Combined with a deterministic inference configuration (sentence splitting,
kana-constrained decoding, beam search width 4), this model **outperforms the
de-facto standard dictionary-based G2P (pyopenjtalk) on all seven evaluation
items of the paper, with statistically significant wins on four** — without
consulting a dictionary at inference time.

| Evaluation item | pyopenjtalk | This system |
|---|---|---|
| jvs overall CER ↓ | 0.933% | **0.924%** |
| Joyo ↑ | 96.54% | **96.96%** (p=2.5e-3) |
| Joyo Parakeet ed. ↑ | 97.01% | **97.68%** (p=2.8e-7) |
| KWDLC-test CER ↓ | 1.236% | **0.934%** (p=1.8e-6) |
| Heteronym set ↑ | 59.2% | **69.1%** (p=3.1e-5) |

Paper: *Surpassing Dictionary-Based G2P for Japanese with a 1B Language Model:
Human-Annotated Training Data and Deterministic Decoding* (arXiv link TBA).
Code & evaluation data: [github.com/tmfi-analytics/slm-ja-g2p](https://github.com/tmfi-analytics/slm-ja-g2p).

## Usage

```python
import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    LogitsProcessor, LogitsProcessorList
from peft import PeftModel

REPO = "t2-fujii/slm-ja-g2p-v11-lora"
BASE = "sbintuitions/sarashina2.2-1b-instruct-v0.1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
         else torch.float16) if dev == "cuda" else torch.float32
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype)
model = PeftModel.from_pretrained(model, REPO).merge_and_unload().to(dev).eval()

SYSTEM = ("あなたは日本語テキストの読み推定器です。与えられた文の読みを、カタカナのみで1行で出力してください。"
          "発音通りに書くこと: 助詞の「は」は「ワ」、「へ」は「エ」、「を」は「オ」。数字や記号も読みをカタカナで書くこと。")
OK = re.compile(r"^[ァ-ヶー、]+$")
allowed = sorted({i for i in range(len(tok)) if OK.match(tok.decode([i]))} | {tok.eos_token_id})
allowed_t = torch.tensor(allowed, device=dev)

class KanaMask(LogitsProcessor):
    def __call__(self, ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, allowed_t] = 0.0
        return scores + mask

SPLIT = re.compile(r"(?<=[。！？])(?![。！？])")

def read(text, num_beams=4):
    outs = []
    for part in (p for p in SPLIT.split(text) if p.strip()):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": part}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(dev)
        out = model.generate(**enc, do_sample=False, num_beams=num_beams,
                             length_penalty=0.0, early_stopping=True,
                             max_new_tokens=min(256, 4 * len(part)),
                             pad_token_id=tok.eos_token_id,
                             logits_processor=LogitsProcessorList([KanaMask()]))
        outs.append(tok.decode(out[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip())
    return "".join(outs)

print(read("著名人が1993年の一日を語った。"))
```

The paper's numbers require this exact configuration (splitting + kana mask +
beam 4). Greedy decoding (`num_beams=1`) is faster and still matches or beats
the dictionary on six of seven items.

## Training

- Base: sarashina2.2-1b-instruct-v0.1, LoRA r=16 (all linear layers), 3 epochs,
  ~2.43M rows.
- Training data: **human-annotated resources and rule-based generation only —
  no LLM-generated data.** Components: Wikipedia sentences double-validated by
  two independent dictionary systems, digit sentences, Aozora Bunko ruby
  (headword-attested), KWDLC train split, ROHAN4600, SudachiDict unique-reading
  words, rule-generated numeral/alphabet sentences, jsut-label.
- The training corpus itself is not redistributed (some sources prohibit it);
  it is **deterministically reproducible** (pinned sources, fixed seeds) with
  the construction notebook in the GitHub repository, which also audits by
  SHA-1 that no evaluation sentence leaks into training.

## License

Adapter weights: MIT (base model sarashina2.2 is MIT).

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
