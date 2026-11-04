"""jvs_nonpara_kana scorer: overall CER, paired 2x2 (exact-match fixes/regressions
vs. pyopenjtalk), and the digit-slice CER (diagnostic).

Usage:
  python eval/eval_jvs.py external/jvs_nonpara_kana/jvs_nonpara_kana.csv <jvs3000_preds.tsv>

Both the model and pyopenjtalk are normalized by the same
pronunciation-equivalence function (paper Sec. 4). The CSV header row is skipped.
"""
import csv, math, os, re, sys
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "ehs", os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_hard_set.py"))
ehs = _ilu.module_from_spec(_spec); _spec.loader.exec_module(ehs)
import jiwer, pyopenjtalk

norm = ehs.norm
DIGIT = re.compile(r"[0-9０-９]")


def mcnemar(b01, b10):
    chi = (abs(b01 - b10) - 1) ** 2 / (b01 + b10) if (b01 + b10) else 0.0
    return math.erfc(math.sqrt(chi / 2))


def main():
    csv_path, pred_path = sys.argv[1], sys.argv[2]
    rows = list(csv.reader(open(csv_path, encoding="utf-8")))
    if rows and rows[0][:1] == ["base"]:
        rows = rows[1:]
    preds = dict(l.rstrip("\n").split("\t", 1)
                 for l in open(pred_path, encoding="utf-8") if "\t" in l)
    tot = e_m = e_p = fix = brk = dtot = derr_m = derr_p = 0
    for r in rows:
        key, text, kana = r[0], r[1], r[2]
        g = norm(kana)
        h = norm(preds.get(key, ""))
        p = norm(pyopenjtalk.g2p(text, kana=True))
        n = len(g); tot += n
        cm, cp = jiwer.cer(g, h) * n, jiwer.cer(g, p) * n
        e_m += cm; e_p += cp
        if DIGIT.search(text):
            dtot += n; derr_m += cm; derr_p += cp
        if p != g and h == g: fix += 1
        if p == g and h != g: brk += 1
    print(f"sentences: {len(rows)}")
    print(f"CER: model {100*e_m/tot:.3f}%  pyopenjtalk {100*e_p/tot:.3f}%")
    print(f"2x2 (exact match): fix {fix} / regress {brk}  net {fix-brk:+d}  "
          f"McNemar p={mcnemar(fix, brk):.3g}")
    print(f"digit slice CER: model {100*derr_m/dtot:.2f}%  "
          f"pyopenjtalk {100*derr_p/dtot:.2f}%  (diagnostic; subset of the above)")


if __name__ == "__main__":
    main()
