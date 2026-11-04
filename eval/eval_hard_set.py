"""辞書不一致困難セットの採点（gold記入前に事前コミット＝分析の事前登録）。

使い方:
  .venv/bin/python experiments/exp03_slm_finetune/eval_hard_set.py \
      data/hard_set/annotation_sheet_v1_filled.tsv \
      [experiments/exp03_slm_finetune/outputs/colab_v6_2/results_v6_2/hardset_preds.tsv ...]

- シート列: id / text / 対象語 / 読み（未記入行・?行は除外として集計）
- gold: 「/」区切りで複数許容。ひらがな可（カタカナへ正規化）
- 判定: 各システムの文全体読み（発音等価正規化）に gold読みのいずれかが含まれるか
- 報告: pyoj誤り側の修正率（95%CI）/ pyoj正解側の破壊率 / 全体ペアMcNemar
"""
import csv, math, re, sys

import os as _os
for _c in ("external/jvs_nonpara_kana",
           _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "external", "jvs_nonpara_kana")):
    if _os.path.exists(_os.path.join(_c, "eval_cer.py")):
        sys.path.insert(0, _c); break
else:
    sys.path.insert(0, "external/jvs_nonpara_kana")
import pyopenjtalk
from eval_cer import normalize_long_vowels
from sudachipy import dictionary as _sd, tokenizer as _st

sud = _sd.Dictionary().create(); SMODE = _st.Tokenizer.SplitMode.C
STRIP = re.compile(r"[。．，！？!?・\s「」『』（）()〔〕［］…‥－—−‐―〜~、]")
KATA_H2K = str.maketrans({chr(c): chr(c + 0x60) for c in range(0x3041, 0x3097)})
VOWEL = {}
for row, v in [("アカガサザタダナハバパマヤラワャァ","a"),("イキギシジチヂニヒビピミリィ","i"),
               ("ウクグスズツヅヌフブプムユルュゥヴ","u"),("エケゲセゼテデネヘベペメレェ","e"),
               ("オコゴソゾトドノホボポモヨロヲョォ","o")]:
    for ch in row: VOWEL[ch] = v

def pron_norm(s):
    s = s.replace("ヲ","オ").replace("ヂ","ジ").replace("ヅ","ズ")
    out = []
    for ch in s:
        if out:
            pv = VOWEL.get(out[-1])
            if ch == "ー" and pv:
                out.append({"a":"ア","i":"イ","u":"ウ","e":"エ","o":"オ"}[pv]); continue
            if (ch == "ウ" and pv in ("o","u")) or (ch == "イ" and pv in ("e","i")):
                out.append({"o":"オ","u":"ウ","e":"エ","i":"イ"}[pv]); continue
        out.append(ch)
    return "".join(out).replace("トユウ", "トイウ")

def norm(s):
    return pron_norm(normalize_long_vowels(STRIP.sub("", s)))

def align_norm(s):
    """アラインメント・照合用の正規化。pron_normの文脈依存規則（オ段+ウ→オ等）は
    区間境界をまたいで作用し、gold単体と全文出力で結果がずれるため適用しない。"""
    return normalize_long_vowels(STRIP.sub("", s).replace("ヲ", "オ"))

def read_pyoj(t):
    return align_norm(pyopenjtalk.g2p(t, kana=True))

def read_sudachi(t):
    out = []
    for m in sud.tokenize(t, SMODE):
        r = m.reading_form()
        if m.part_of_speech()[0] == "助詞":
            if m.surface() == "は": r = "ワ"
            elif m.surface() == "へ": r = "エ"
        out.append(r)
    return align_norm("".join(out))

def aligned_span(ref, hyp, s, e):
    """ref[s:e]に対応するhypの区間を編集距離アラインメントで抽出（公式eval_dataset.py方式のG2P適用）"""
    n, m = len(ref), len(hyp)
    D = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): D[i][0] = i
    for j in range(m+1): D[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+(ref[i-1] != hyp[j-1]))
    # backtrace: ref位置→hyp位置の対応
    i, j = n, m
    map_end = [m]*(n+1)
    while i > 0 or j > 0:
        map_end[i] = j
        if i > 0 and j > 0 and D[i][j] == D[i-1][j-1]+(ref[i-1] != hyp[j-1]):
            i, j = i-1, j-1
        elif i > 0 and D[i][j] == D[i-1][j]+1:
            i -= 1
        else:
            j -= 1
    map_end[0] = 0
    return hyp[map_end[s]:map_end[e]]

def judge_aligned(text, target, golds, hyp_read):
    """参照読み（対象語をgoldで固定した合成参照）とのアラインで対象区間を採点。
    アラインはalign_norm空間で行い、最終比較のみ発音等価（pron_norm）で行う。
    golds/hyp_readはalign_norm済みを渡すこと。"""
    i = text.find(target)
    if i < 0: return None
    pre = align_norm(pyopenjtalk.g2p(text[:i], kana=True)) if text[:i].strip("、") else ""
    post = align_norm(pyopenjtalk.g2p(text[i+len(target):], kana=True)) if text[i+len(target):].strip("、") else ""
    for g in golds:
        ref = pre + g + post
        span = aligned_span(ref, hyp_read, len(pre), len(pre)+len(g))
        if pron_norm(span) == pron_norm(g):
            return True
    return False

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*(c-h), 100*(c+h))

def mcnemar(b01, b10):
    chi = (abs(b01-b10)-1)**2/(b01+b10) if (b01+b10) else 0.0
    return math.erfc(math.sqrt(chi/2))

def main():
    sheet = sys.argv[1]
    pred_files = sys.argv[2:]
    delim = "," if sheet.endswith(".csv") else "\t"
    rows = list(csv.reader(open(sheet, encoding="utf-8"), delimiter=delim))[1:]
    items = []
    n_skip = 0
    KANA_ONLY = re.compile(r"^[ァ-ヶー]+$")
    n_invalid = 0
    for r in rows:
        if len(r) < 4 or not r[3].strip() or r[3].strip() == "?":
            n_skip += 1; continue
        golds = {align_norm(g.strip().translate(KATA_H2K)) for g in re.split(r"[,、/／・]", r[3]) if g.strip()}
        golds = {g for g in golds if g}
        if not golds or any(not KANA_ONLY.match(g) for g in golds):
            n_invalid += 1
            print(f"  形式不正で除外: {r[0]} {r[2]} -> {r[3]!r}")
            continue
        items.append({"id": r[0], "text": r[1], "target": r[2], "golds": golds})
    print(f"採点対象 {len(items)} 文（未記入/?で除外 {n_skip}, 形式不正 {n_invalid}）")

    systems = {"pyopenjtalk": lambda t: read_pyoj(t), "sudachi": lambda t: read_sudachi(t)}
    preds = {}
    for pf in pred_files:
        parts = pf.split("/")
        name = parts[-1].replace("_preds.tsv", "").replace(".tsv", "")
        if name == "hardset" and len(parts) >= 2:
            name = parts[-2] + "/hardset"   # results_v5/hardset 等（同名衝突の回避）
        d = dict(l.rstrip("\n").split("\t", 1) for l in open(pf, encoding="utf-8") if "\t" in l)
        preds[name] = d
        systems[name] = lambda t, _d=d, _it=None: None  # placeholder; handled below

    # 全システムの正誤表（主指標=アラインメント区間採点、副=部分一致で頑健性確認）
    ok = {name: {} for name in systems}
    ok_sub = {name: {} for name in systems}
    for it in items:
        reads = {"pyopenjtalk": read_pyoj(it["text"]), "sudachi": read_sudachi(it["text"])}
        for name, d in preds.items():
            reads[name] = align_norm(d.get(it["id"], ""))
        for name, r in reads.items():
            a = judge_aligned(it["text"], it["target"], it["golds"], r)
            sub = any(g in r for g in it["golds"])
            ok[name][it["id"]] = sub if a is None else a
            ok_sub[name][it["id"]] = sub

    pw = [it for it in items if not ok["pyopenjtalk"][it["id"]]]   # pyoj誤り側
    pr = [it for it in items if ok["pyopenjtalk"][it["id"]]]       # pyoj正解側
    print(f"pyoj誤り側 {len(pw)} 文 / pyoj正解側 {len(pr)} 文")
    for name in ok:
        acc = sum(ok[name][it["id"]] for it in items)
        fx = sum(ok[name][it["id"]] for it in pw)
        br = sum(not ok[name][it["id"]] for it in pr)
        lo, hi = wilson_ci(fx, len(pw))
        b01 = sum(ok[name][it["id"]] and not ok["pyopenjtalk"][it["id"]] for it in items)
        b10 = sum((not ok[name][it["id"]]) and ok["pyopenjtalk"][it["id"]] for it in items)
        p = mcnemar(b01, b10)
        n_dis = sum(ok[name][it["id"]] != ok_sub[name][it["id"]] for it in items)
        print(f"{name}: 全体 {acc}/{len(items)} ({100*acc/len(items):.1f}%)  "
              f"修正 {fx}/{len(pw)} ({100*fx/max(len(pw),1):.1f}%, CI {lo:.0f}-{hi:.0f})  "
              f"破壊 {br}/{len(pr)} ({100*br/max(len(pr),1):.1f}%)  "
              f"vs pyoj 勝{b01}/負{b10} p={p:.3g}  [部分一致との判定差 {n_dis}件]")
    # 明細（証跡用）
    out_tsv = sheet.replace(".tsv", "_scored.tsv")
    with open(out_tsv, "w", encoding="utf-8") as f:
        names = list(ok)
        f.write("id\ttext\t対象語\tgold\t" + "\t".join(names) + "\n")
        for it in items:
            f.write(f'{it["id"]}\t{it["text"]}\t{it["target"]}\t{"/".join(sorted(it["golds"]))}\t'
                    + "\t".join("o" if ok[n][it["id"]] else "x" for n in names) + "\n")
    print("明細:", out_tsv)

if __name__ == "__main__":
    main()
