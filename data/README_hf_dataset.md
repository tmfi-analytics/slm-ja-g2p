---
license: cc-by-sa-4.0
language:
- ja
task_categories:
- text2text-generation
tags:
- japanese
- g2p
- heteronym
- reading-estimation
- evaluation
size_categories:
- n<1K
---

# Japanese Heteronym Reading Evaluation Set v1

300 real Japanese Wikipedia sentences for evaluating context-dependent readings
of **common words** on which dictionary-based G2P errs (e.g., 行く/行う,
通る/通う, 研究所 -jo/-sho, 世論, 重複). Selection is fully mechanical
(top-300 pyopenjtalk×Sudachi disagreement words, one sentence per word, fixed
seed); gold readings were blind-annotated by the author (265 valid items).
pyopenjtalk scores 59.2% on this set.

Columns: `id`, `text`, `target_word`, `gold_readings` (katakana; multiple
acceptable readings separated by `/`; empty = excluded from scoring).

Construction details, scoring script, and the accompanying paper:
see the GitHub repository:
[github.com/tmfi-analytics/slm-ja-g2p](https://github.com/tmfi-analytics/slm-ja-g2p).

License: sentences from Japanese Wikipedia (CC BY-SA 4.0); annotations released
under the same license.

```bibtex
@misc{fujii2026slmg2p,
  title  = {Surpassing Dictionary-Based G2P for Japanese with a 1B Language
            Model: Human-Annotated Training Data and Deterministic Decoding},
  author = {Fujii, Takafumi},
  year   = {2026},
  note   = {arXiv preprint (ID TBA)}
}
```
