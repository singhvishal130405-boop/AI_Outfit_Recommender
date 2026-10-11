import pandas as pd, numpy as np, random
import json
label_maps = json.load(open("work/label_maps.json"))
INV = {h: {i: c for c, i in label_maps[h].items()} for h in ["usage", "season"]}

def load_split(split):
    df = pd.read_csv(f"work/{split}.csv")
    df["usage"] = df["usage"].map(INV["usage"])      # ints -> "Casual", "Formal", ...
    df["season"] = df["season"].map(INV["season"])   # ints -> "Summer", "Winter", ...
    assert df["usage"].isin(["Casual", "Sports", "Ethnic", "Formal"]).all()
    return df

random.seed(42); np.random.seed(42)

NEUTRALS = {"Black", "White", "Grey", "Navy Blue", "Beige", "Brown"}
GOOD_COLORS = {frozenset(p) for p in [
    ("Blue", "Yellow"), ("Blue", "Purple"), ("Blue", "Green"),
    ("Pink", "Purple"), ("Green", "Yellow"), ("Pink", "Blue")]}
SOFT_NEUTRAL_PAIRS = {frozenset(("Navy Blue", "Black")): 0.7,
                      frozenset(("Brown", "Black")): 0.8}

VALID_SLOTS = {frozenset(p) for p in [
    ("top", "bottom"), ("top", "footwear"),
    ("bottom", "footwear"), ("dress", "footwear")]}

USAGE = {  # symmetric usage compatibility
    ("Casual", "Casual"): 1.0, ("Formal", "Formal"): 1.0,
    ("Sports", "Sports"): 1.0, ("Ethnic", "Ethnic"): 1.0,
    ("Casual", "Sports"): 0.6, ("Casual", "Formal"): 0.5,
    ("Casual", "Ethnic"): 0.6, ("Formal", "Ethnic"): 0.5,
    ("Formal", "Sports"): 0.05, ("Ethnic", "Sports"): 0.1,
}
def usage_score(u1, u2):
    return USAGE.get((u1, u2), USAGE.get((u2, u1), 0.3))

CLASHES = {frozenset(p) for p in [
    ("Formal Shoes", "Shorts"), ("Formal Shoes", "Track Pants"),
    ("Sports Shoes", "Kurtas"), ("Sports Shoes", "Dresses"),
    ("Heels", "Track Pants"), ("Kurtas", "Shorts"), ("Kurtas", "Track Pants"),
    ("Flip Flops", "Formal Shoes")]}

def color_score(c1, c2):
    if c1 in NEUTRALS or c2 in NEUTRALS:
        return SOFT_NEUTRAL_PAIRS.get(frozenset((c1, c2)), 1.0)
    if c1 == c2: return 0.7
    return 0.9 if frozenset((c1, c2)) in GOOD_COLORS else 0.35

def compat_score(a, b):
    if frozenset((a.slot, b.slot)) not in VALID_SLOTS: return 0.0
    if a.gender != b.gender and "Unisex" not in (a.gender, b.gender): return 0.0
    s = usage_score(a.usage, b.usage) * color_score(a.baseColour, b.baseColour)
    if {a.season, b.season} == {"Summer", "Winter"}: s *= 0.7
    if frozenset((a.articleType, b.articleType)) in CLASHES: s *= 0.2
    return s

def build_pairs(df, n_draws):
    rows = list(df.itertuples(index=False))
    out = []
    for _ in range(n_draws):
        a, b = random.sample(rows, 2)
        s = compat_score(a, b)
        s = float(np.clip(s + np.random.normal(0, 0.05), 0, 1)) if s > 0 else 0.0
        out.append((a.id, b.id, s))
    p = pd.DataFrame(out, columns=["id_a", "id_b", "score"])
    p["label"] = (p["score"] >= 0.6).astype(int)
    # balance: keep all positives, an equal number of negatives
    pos = p[p.label == 1]
    neg = p[p.label == 0].sample(n=min(len(pos), (p.label == 0).sum()), random_state=42)
    return pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

for split, draws in [("train", 400_000), ("val", 50_000), ("test", 50_000)]:
    df = load_split(split)
    pairs = build_pairs(df, draws)
    pairs.to_csv(f"work/pairs_{split}.csv", index=False)
    print(f"{split}: {len(pairs)} pairs | positives={pairs.label.sum()} "
          f"| mean score={pairs.score.mean():.2f}")

# sanity check: show a few scored examples with readable names
meta = load_split("train").set_index("id")
p = pd.read_csv("work/pairs_train.csv")
cols = ["articleType", "baseColour", "usage", "gender"]
print("\n--- sample pairs ---")
for _, r in p.sample(12, random_state=1).iterrows():
    a, b = meta.loc[int(r.id_a), cols].tolist(), meta.loc[int(r.id_b), cols].tolist()
    print(f"{r.score:.2f}  {a}  +  {b}")