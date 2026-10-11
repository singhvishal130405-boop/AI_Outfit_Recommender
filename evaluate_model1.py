import tensorflow as tf, pandas as pd, numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

IMG_SIZE, BATCH = 128, 64
HEADS = ["article", "color", "usage", "season"]
AUTOTUNE = tf.data.AUTOTUNE

label_maps = json.load(open("work/label_maps.json"))
class_names = {h: [c for c, i in sorted(label_maps[h].items(), key=lambda kv: kv[1])]
               for h in HEADS}

test_df = pd.read_csv("work/test.csv")

def load(p):
    img = tf.io.read_file(p)
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

ds = (tf.data.Dataset.from_tensor_slices(test_df["path"].values)
        .map(load, num_parallel_calls=AUTOTUNE)
        .batch(BATCH).prefetch(AUTOTUNE))     # no shuffle: order must match the csv

model1 = tf.keras.models.load_model("work/model1.keras")
preds = model1.predict(ds, verbose=0)

print("\n=== TEST ACCURACY ===")
for h in HEADS:
    y_true = test_df[h].values
    y_pred = preds[h].argmax(axis=1)
    print(f"{h:8s} {accuracy_score(y_true, y_pred):.3f}")

# per-class report + confusion matrix image for article and color
for h in ["article", "color"]:
    y_true = test_df[h].values
    y_pred = preds[h].argmax(axis=1)
    names = class_names[h]
    labels = list(range(len(names)))

    if h == "article":
        print("\n=== ARTICLE: per-class report ===")
        print(classification_report(y_true, y_pred, labels=labels,
                                    target_names=names, digits=2, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels); ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(labels); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{h} confusion matrix (row-normalised, test set)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(f"work/confusion_{h}.png", dpi=130)
    plt.close(fig)

    # top confusions in plain text
    off = [(cm[i, j], names[i], names[j]) for i in labels for j in labels if i != j]
    off.sort(reverse=True)
    print(f"\n=== {h.upper()}: top 8 confusions (count, true -> predicted) ===")
    for n, t, p in off[:8]:
        print(f"{n:4d}  {t} -> {p}")

print("\nSaved: work/confusion_article.png, work/confusion_color.png")