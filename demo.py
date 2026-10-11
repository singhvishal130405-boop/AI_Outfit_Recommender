import tensorflow as tf, pandas as pd, numpy as np, json, argparse, itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ap = argparse.ArgumentParser()
ap.add_argument("--image", help="path to a query image (default: random test item)")
ap.add_argument("--gender", choices=["Men", "Women", "Unisex"],
                help="required with --image")
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--out", default="work/demo.png")
args = ap.parse_args()
if args.image and not args.gender:
    ap.error("--gender is required when using --image")

HEADS = ["article", "color", "usage", "season"]
GENDERS = {"Men": 0, "Women": 1, "Unisex": 2}
SLOT_MAP = {
    "Tshirts":"top","Shirts":"top","Tops":"top","Kurtas":"top","Sweatshirts":"top",
    "Jackets":"top","Sweaters":"top","Jeans":"bottom","Trousers":"bottom",
    "Shorts":"bottom","Track Pants":"bottom","Skirts":"bottom","Leggings":"bottom",
    "Dresses":"dress","Casual Shoes":"footwear","Sports Shoes":"footwear",
    "Formal Shoes":"footwear","Heels":"footwear","Flats":"footwear",
    "Sandals":"footwear","Flip Flops":"footwear"}
PARTNERS = {"top": ["bottom", "footwear"], "bottom": ["top", "footwear"],
            "dress": ["footwear"], "footwear": ["top", "bottom", "dress"]}

label_maps = json.load(open("work/label_maps.json"))
names = {h: [c for c, _ in sorted(label_maps[h].items(), key=lambda kv: kv[1])] for h in HEADS}

model1 = tf.keras.models.load_model("work/model1.keras")
model2 = tf.keras.models.load_model("work/model2.keras")

# ---- gallery = test split with cached Model 1 features ----
gal = pd.read_csv("work/test.csv")
feats = np.load("work/feats_test.npz")
assert (feats["ids"] == gal["id"].values).all()
G = feats["pred"]

# ---- pick the query ----
exclude = None
if args.image:
    q_path, q_gender, true_info = args.image, args.gender, None
else:
    rng = np.random.default_rng(args.seed)
    exclude = int(rng.integers(len(gal)))
    row = gal.iloc[exclude]
    q_path, q_gender = row["path"], row["gender"]
    true_info = f"{row['articleType']} | {row['baseColour']}"

def load_img(p):
    img = tf.io.read_file(p)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    return tf.image.resize(img, (128, 128))

# ---- Model 1: image -> attributes ----
pred = model1.predict(load_img(q_path)[None], verbose=0)
top = {h: int(pred[h][0].argmax()) for h in HEADS}
conf = {h: float(pred[h][0].max()) for h in HEADS}
q_article = names["article"][top["article"]]
q_slot = SLOT_MAP[q_article]
q_vec = np.concatenate([pred[h][0] for h in HEADS] + [np.eye(3)[GENDERS[q_gender]]]).astype("float32")

q_desc = (f"{q_article} ({conf['article']:.0%})\n{names['color'][top['color']]} | "
          f"{names['usage'][top['usage']]} | {q_gender}")
print("\nQUERY:", q_path)
print("Model 1 predicts:", q_desc.replace("\n", " | "), f"| slot={q_slot}")
if true_info:
    print("Actual label    :", true_info)

# ---- Model 2: score the query against gallery items in partner slots ----
ok = np.ones(len(gal), bool) if q_gender == "Unisex" else np.array(gal["gender"].isin([q_gender, "Unisex"]), dtype=bool)
if exclude is not None:
    ok[exclude] = False

def score(a_vecs, b_vecs):
    return model2.predict([a_vecs, b_vecs], batch_size=2048, verbose=0).ravel()

recs = {}
for slot in PARTNERS[q_slot]:
    idx = np.where(ok & (gal["slot"].values == slot))[0]
    if len(idx) == 0:          # e.g. no men's dresses in the gallery
        continue
    s = score(np.repeat(q_vec[None], len(idx), 0), G[idx])
    order = np.argsort(-s)[:10]
    recs[slot] = [(idx[i], float(s[i])) for i in order]

# ---- best full outfit (3 pieces) when the query has two partner slots ----
outfit = None
OUTFIT = {"top": ("bottom", "footwear"), "bottom": ("top", "footwear"),
          "footwear": ("top", "bottom")}

if q_slot in OUTFIT and all(s in recs for s in OUTFIT[q_slot]):
    s1, s2 = OUTFIT[q_slot]
    combos = list(itertools.product(recs[s1], recs[s2]))
    ab = score(np.stack([G[a[0]] for a, b in combos]), np.stack([G[b[0]] for a, b in combos]))
    totals = [(a[1] + b[1] + ab[k]) / 3 for k, (a, b) in enumerate(combos)]
    k = int(np.argmax(totals))
    outfit = (combos[k][0], combos[k][1], float(totals[k]))

def label(i):
    r = gal.iloc[i]
    return f"{r['articleType']} | {r['baseColour']}"

print()
for slot, lst in recs.items():
    print(f"Top {slot} picks:")
    for i, s in lst[:3]:
        print(f"   {s:.2f}  {label(i)}")
if outfit:
    print(f"\nBest full outfit (score {outfit[2]:.2f}): {q_article} + "
          f"{label(outfit[0][0])} + {label(outfit[1][0])}")

# ---- plot ----
rows = len(recs) + (1 if outfit else 0)
fig, axes = plt.subplots(rows, 4, figsize=(11, 3.2 * rows), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")
qimg = mpimg.imread(q_path)

for r, (slot, lst) in enumerate(recs.items()):
    axes[r, 0].imshow(qimg)
    axes[r, 0].set_title(f"QUERY -> best {slot}\n{q_desc}", fontsize=8, color="navy")
    for c, (i, s) in enumerate(lst[:3], start=1):
        axes[r, c].imshow(mpimg.imread(gal.iloc[i]["path"]))
        axes[r, c].set_title(f"{label(i)}\nscore {s:.2f}", fontsize=8)

if outfit:
    r = rows - 1
    axes[r, 0].imshow(qimg); axes[r, 0].set_title("BEST OUTFIT\nquery piece", fontsize=8, color="darkgreen")
    for c, (i, _) in enumerate([outfit[0], outfit[1]], start=1):
        axes[r, c].imshow(mpimg.imread(gal.iloc[i]["path"]))
        axes[r, c].set_title(label(i), fontsize=8)
    axes[r, 3].text(0.5, 0.5, f"outfit score\n{outfit[2]:.2f}", ha="center", va="center", fontsize=14)

fig.suptitle("AI outfit recommender: Model 1 (image -> attributes) + Model 2 (compatibility)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(args.out, dpi=130)
print("\nSaved:", args.out)
plt.show()