# Heteronym Evaluation Set v1 — Construction Procedure

File: `heteronym_eval_set_v1.csv` (CSV; columns: `id`, `text`, `target_word`,
`gold_readings` (katakana; multiple acceptable readings separated by `/`;
empty = excluded from scoring)). 300 rows; 265 were valid for scoring after excluding
unfilled/uncertain items (see paper §4).

## Design goal

Evaluate **context-dependent readings of common words** on which dictionary
G2P errs — not rare or literary readings. No public evaluation set with real
sentences and human labels focuses on this class.

## Procedure (fully mechanical selection; blind annotation)

1. **Candidate pool**: 2.6M Japanese Wikipedia sentences (snapshot 20231101.ja),
   sentence-split, 10–60 chars, containing kanji.
2. **Disagreement mining**: extract the 678k sentences where pyopenjtalk
   (naist-jdic) and Sudachi (SudachiDict, split mode C) produce different
   readings. Rationale: sentences where two independent dictionaries disagree
   are, by an objective criterion, sentences where the reading judgment is hard.
3. **Word selection**: rank the disagreement-causing words by frequency; take
   the **top 300 common words** (proper nouns and numerals excluded).
4. **Sentence sampling**: for each word, sample exactly one containing sentence
   with a fixed random seed (42).
5. **Blind annotation**: the author annotated the gold reading of the target
   word from sentence context **before any fine-tuned model was evaluated on
   the set**; items where the reading could not be determined were marked and
   excluded (265 valid). Multiple readings are listed when genuinely
   acceptable.

Because steps 1–4 are entirely mechanical, cherry-picking sentences favorable
to any system is structurally impossible. The set was excluded from all
training-data construction by exact match, verified by a SHA-1 audit over
every training row.

## Scoring

`eval/eval_hard_set.py` judges the target-word span by character-level
alignment of the whole-sentence reading against a reference in which the
target span is fixed to the gold reading, with pronunciation-equivalence
normalization applied identically to every system (see paper §4).

Reference numbers on the 265 valid items:

| System | Accuracy | fixes / regressions vs. pyopenjtalk |
|---|---|---|
| pyopenjtalk 0.4.1 (naist-jdic 1.11) | 59.2% | — |
| Sudachi (SudachiDict 20260723) | 73.6% | 99 / 61 |
| Proposed system (paper) | 69.1% | 31 / 5 |

License: sentences from Japanese Wikipedia, CC BY-SA 4.0; annotations released
under the same license.
