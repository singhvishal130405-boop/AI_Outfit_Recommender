import pandas as pd, numpy as np, os, json
from sklearn.model_selection import train_test_split

DATA_DIR = os.environ.get("DATA_DIR", "dataset")
CSV_PATH = os.path.join(DATA_DIR, "styles.csv")
IMG_DIR  = os.path.join(DATA_DIR, "images")

df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
print("loaded:", df.shape)

# 1) keep only clothing + footwear item types, and map each to an outfit "slot"
SLOT_MAP = {
    # tops
    "Tshirts":"top","Shirts":"top","Tops":"top","Kurtas":"top","Sweatshirts":"top",
    "Jackets":"top","Sweaters":"top",
    # bottoms
    "Jeans":"bottom","Trousers":"bottom","Shorts":"bottom","Track Pants":"bottom",
    "Skirts":"bottom","Leggings":"bottom",
    # one-piece
    "Dresses":"dress",
    # footwear
    "Casual Shoes":"footwear","Sports Shoes":"footwear","Formal Shoes":"footwear",
    "Heels":"footwear","Flats":"footwear","Sandals":"footwear","Flip Flops":"footwear",
}
df = df[df["articleType"].isin(SLOT_MAP)].copy()
df["slot"] = df["articleType"].map(SLOT_MAP)

# 2) drop adult-only simplification: remove kids, keep Men/Women/Unisex
df = df[df["gender"].isin(["Men","Women","Unisex"])]

# 3) keep only the usable usage values
df = df[df["usage"].isin(["Casual","Sports","Ethnic","Formal"])]

# 4) drop missing season/colour, keep the 12 most common colours
df = df.dropna(subset=["season","baseColour"])
top_colors = df["baseColour"].value_counts().head(12).index
df = df[df["baseColour"].isin(top_colors)]

# 5) drop article types with too few samples
vc = df["articleType"].value_counts()
df = df[df["articleType"].isin(vc[vc >= 200].index)]

# 6) drop rows whose image file is missing
df["path"] = df["id"].astype(str).map(lambda i: os.path.join(IMG_DIR, i + ".jpg"))
df = df[df["path"].map(os.path.exists)].reset_index(drop=True)
print("after cleaning:", df.shape)

# 7) encode each target as an integer
TARGETS = {"article":"articleType","color":"baseColour","usage":"usage","season":"season"}
label_maps = {}
for name, col in TARGETS.items():
    classes = sorted(df[col].unique())
    label_maps[name] = {c:i for i,c in enumerate(classes)}
    df[name] = df[col].map(label_maps[name])
    print(f"{name}: {len(classes)} classes ->", classes)

# 8) train / val / test split (80/10/10), stratified on articleType
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["article"])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["article"])
print(len(train_df), len(val_df), len(test_df))

# 9) save everything for the next step
os.makedirs("work", exist_ok=True)
train_df.to_csv("work/train.csv", index=False)
val_df.to_csv("work/val.csv", index=False)
test_df.to_csv("work/test.csv", index=False)
json.dump(label_maps, open("work/label_maps.json","w"), indent=2)