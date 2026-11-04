"""タグ区間アラインメント採点（EVAL_PROTOCOL「指標の割り当て」の実装）。
Joyo/Parakeet/g2pbench/KWDLC-test を、任意の予測TSVに対して採点する。

使い方:
  eval_aligned_bench.py joyo <preds.tsv> [<preds2.tsv> ...]
  eval_aligned_bench.py parakeet <preds.tsv> ...   (キーは pk{i:05d})
  eval_aligned_bench.py g2pbench <preds.tsv> ...
  eval_aligned_bench.py kwdlc <gold.json> <preds.tsv> ...   (キーは kw{i:04d})
各ベンチでpyopenjtalkも同一判定器で採点し、ペアMcNemarを出す。
"""
import csv, json, math, re, sys

sys.path.insert(0, "external/jvs_nonpara_kana")
import importlib.util as _ilu

import os
_spec = _ilu.spec_from_file_location("ehs", os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_hard_set.py"))
ehs = _ilu.module_from_spec(_spec); _spec.loader.exec_module(ehs)
import pyopenjtalk

align_norm, pron_norm, aligned_span = ehs.align_norm, ehs.pron_norm, ehs.aligned_span
KATA_H2K = ehs.KATA_H2K

def load_preds(fn):
    return dict(l.rstrip("\n").split("\t", 1) for l in open(fn, encoding="utf-8") if "\t" in l)

def mcnemar(b01, b10):
    chi = (abs(b01 - b10) - 1) ** 2 / (b01 + b10) if (b01 + b10) else 0.0
    return math.erfc(math.sqrt(chi / 2))

def judge_span(pre_text, target_text, post_text, golds, hyp_align):
    """参照= pyoj(pre)+gold+pyoj(post) とのアラインで対象区間を採点（align空間）"""
    pre = align_norm(pyopenjtalk.g2p(pre_text, kana=True)) if pre_text.strip("、。 ") else ""
    post = align_norm(pyopenjtalk.g2p(post_text, kana=True)) if post_text.strip("、。 ") else ""
    for g in golds:
        ref = pre + g + post
        span = aligned_span(ref, hyp_align, len(pre), len(pre) + len(g))
        if pron_norm(span) == pron_norm(g):
            return True
    return False

def report(name, cases, systems):
    """cases: [(id, ok判定関数へ渡す引数…)] 形式は呼び出し側で処理済みの ok辞書"""
    base = systems["pyopenjtalk"]
    n = len(next(iter(systems.values())))
    for sname, ok in systems.items():
        acc = sum(ok.values())
        b01 = sum(ok[k] and not base[k] for k in ok)
        b10 = sum((not ok[k]) and base[k] for k in ok)
        p = mcnemar(b01, b10)
        extra = "" if sname == "pyopenjtalk" else f"  vs pyoj 勝{b01}/負{b10} p={p:.3g}"
        print(f"{name} {sname}: {acc}/{n} ({100*acc/n:.3f}%){extra}")

def _judge_tagged(items, pred_files):
    systems = {"pyopenjtalk": {}}
    preds = {fn: load_preds(fn) for fn in pred_files}
    for it in items:
        s, e, ref = it["s"], it["e"], it["ref"]
        gold_span = ref[s:e]
        accept = {pron_norm(gold_span)} | {pron_norm(a) for a in it["accepts"]}
        for name, hyp_raw in [("pyopenjtalk", pyopenjtalk.g2p(it["text"], kana=True))] + \
                             [(fn, d.get(it["key"], "")) for fn, d in preds.items()]:
            hyp = align_norm(hyp_raw)
            span = aligned_span(ref, hyp, s, e)
            systems.setdefault(name, {})[it["key"]] = pron_norm(span) in accept
    return systems

def eval_tagged(bench, pred_files):
    from datasets import load_dataset
    items = []
    if bench in ("joyo", "parakeet"):
        if bench == "joyo":
            rows = [(ex["key"], ex["normalized_text"], ex["normalized_pron"], [])
                    for ex in load_dataset("sbintuitions/joyo-kanji-yomi-benchmark", split="train")]
        else:
            rows = [(f"pk{i:05d}", ex["text"], ex["tagged_yomi"],
                     [r for r in ex["readings"]["natural"] + ex["readings"]["marginal"] if r])
                    for i, ex in enumerate(load_dataset("Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet", split="test"))]
        for key, text, pron, accepts in rows:
            m = re.search(r"<([^<>]+)>", pron)
            if not m: continue
            pre_raw = pron[:m.start()].replace("<", "").replace(">", "")
            s = len(align_norm(pre_raw))
            e = len(align_norm(pre_raw + m.group(1)))
            ref = align_norm(pron.replace("<", "").replace(">", ""))
            items.append({"key": key, "text": text, "s": s, "e": e, "ref": ref,
                          "accepts": [align_norm(a) for a in accepts]})
        systems = _judge_tagged(items, pred_files)
        report(bench, items, systems)
        return items, systems
    if bench == "g2pbench":
        items = []
        for it in [json.loads(l) for l in open("external/ja_tts_g2p_bench/items_read_bench_v1.jsonl", encoding="utf-8")]:
            text = it["surface"]
            i = text.find(it["target"])
            if i < 0: continue
            items.append({"key": it["id"], "text": text, "pre": text[:i], "tgt": it["target"],
                          "post": text[i+len(it["target"]):],
                          "golds": [align_norm(it["reading_expected"].translate(KATA_H2K))],
                          "category": it["category"]})
    else:
        raise SystemExit(f"unknown bench {bench}")
    systems = {"pyopenjtalk": {}}
    preds = {fn: load_preds(fn) for fn in pred_files}
    for it in items:
        hyp_p = align_norm(pyopenjtalk.g2p(it["text"], kana=True))
        # pre/postはテキスト由来なのでpyoj読みに変換して判定
        systems["pyopenjtalk"][it["key"]] = judge_span(it["pre"], it["tgt"], it["post"], it["golds"], hyp_p)
        for fn, d in preds.items():
            systems.setdefault(fn, {})[it["key"]] = judge_span(
                it["pre"], it["tgt"], it["post"], it["golds"], align_norm(d.get(it["key"], "")))
    report(bench, items, systems)
    return items, systems

def eval_kwdlc(gold_json, pred_files):
    import jiwer
    from eval_cer import normalize_long_vowels
    def norm(s): return pron_norm(normalize_long_vowels(ehs.STRIP.sub("", s)))
    gold = json.load(open(gold_json, encoding="utf-8"))
    preds = {fn: load_preds(fn) for fn in pred_files}
    for fn, d in [("pyopenjtalk", None)] + list(preds.items()):
        nc = ne = 0; win = lose = 0
        for i, (t, kana) in enumerate(gold):
            g = norm(kana)
            if not g: continue
            hyp = norm(pyopenjtalk.g2p(t, kana=True)) if d is None else norm(d.get(f"kw{i:04d}", ""))
            c = jiwer.cer(g, hyp); nc += len(g); ne += c * len(g)
            cp = jiwer.cer(g, norm(pyopenjtalk.g2p(t, kana=True)))
            if c < cp: win += 1
            elif c > cp: lose += 1
        extra = "" if d is None else f"  勝{win}/負{lose} p={mcnemar(win, lose):.3g}"
        print(f"kwdlc {fn}: CER={100*ne/nc:.3f}%{extra}")

if __name__ == "__main__":
    bench = sys.argv[1]
    if bench == "kwdlc":
        eval_kwdlc(sys.argv[2], sys.argv[3:])
    else:
        eval_tagged(bench, sys.argv[2:])
