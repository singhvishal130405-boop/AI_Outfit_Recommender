import tensorflow as tf, pandas as pd, numpy as np, json, os
from tensorflow.keras import layers, Model
from sklearn.metrics import roc_auc_score, accuracy_score

IMG_SIZE, BATCH = 128, 64
HEADS = ["article", "color", "usage", "season"]
GENDERS = {"Men": 0, "Women": 1, "Unisex": 2}
AUTOTUNE = tf.data.AUTOTUNE

label_maps = json.load(open("work/label_maps.json"))
n_cls = {h: len(label_maps[h]) for h in HEADS}
model1 = tf.keras.models.load_model("work/model1.keras")

def load_img(p):
    img = tf.io.read_file(p)
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

def get_features(split):
    cache = f"work/feats_{split}.npz"
    if os.path.exists(cache):
        return dict(np.load(cache))
    df = pd.read_csv(f"work/{split}.csv")
    ds = (tf.data.Dataset.from_tensor_slices(df["path"].values)
            .map(load_img, num_parallel_calls=AUTOTUNE)
            .batch(BATCH).prefetch(AUTOTUNE))
    preds = model1.predict(ds, verbose=0)
    pred_probs = np.concatenate([preds[h] for h in HEADS], axis=1)
    gt = np.concatenate([np.eye(n_cls[h])[df[h].values] for h in HEADS], axis=1)
    g = np.eye(3)[df["gender"].map(GENDERS).values]
    out = {"ids": df["id"].values,
           "pred": np.concatenate([pred_probs, g], axis=1).astype("float32"),
           "gt":   np.concatenate([gt, g], axis=1).astype("float32")}
    np.savez(cache, **out)
    return out

def get_pairs(split, kind):
    f = get_features(split)
    pos = {i: k for k, i in enumerate(f["ids"])}
    p = pd.read_csv(f"work/pairs_{split}.csv")
    ia, ib = p["id_a"].map(pos).values, p["id_b"].map(pos).values
    X = f[kind]
    # "hard" = slot/gender-valid pairs (score > 0): all positives + the tricky negatives
    return X[ia], X[ib], p["label"].values.astype("float32"), (p["score"].values > 0)

print("extracting features (first run only)...")
Xa_tr, Xb_tr, y_tr, _ = get_pairs("train", "pred")
Xa_va, Xb_va, y_va, _ = get_pairs("val", "pred")
feat_dim = Xa_tr.shape[1]
print("feature dim:", feat_dim, "| train pairs:", len(y_tr))

# --- Model 2: shared encoder + symmetric combiner ---
inp = layers.Input((feat_dim,))
e = layers.Dense(64, activation="relu")(inp)
e = layers.Dense(64, activation="relu")(e)
encoder = Model(inp, e, name="item_encoder")

a_in, b_in = layers.Input((feat_dim,)), layers.Input((feat_dim,))
za, zb = encoder(a_in), encoder(b_in)
diff = layers.Subtract()([za, zb])
h = layers.Concatenate()([layers.Add()([za, zb]),        # symmetric terms only,
                          layers.Multiply()([za, zb]),   # so score(A,B) == score(B,A)
                          layers.Multiply()([diff, diff])])
h = layers.Dense(128, activation="relu")(h)
h = layers.Dropout(0.3)(h)
h = layers.Dense(64, activation="relu")(h)
out = layers.Dense(1, activation="sigmoid")(h)
model2 = Model([a_in, b_in], out, name="compat_model")

model2.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss="binary_crossentropy",
               metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

model2.fit([Xa_tr, Xb_tr], y_tr, validation_data=([Xa_va, Xb_va], y_va),
           epochs=20, batch_size=512, verbose=2,
           callbacks=[tf.keras.callbacks.EarlyStopping(
               monitor="val_auc", mode="max", patience=3, restore_best_weights=True)])
model2.save("work/model2.keras")

# --- final evaluation on the untouched test pairs ---
def report(name, kind):
    Xa, Xb, y, hard = get_pairs("test", kind)
    p = model2.predict([Xa, Xb], batch_size=2048, verbose=0).ravel()
    print(f"{name:34s} AUC={roc_auc_score(y, p):.3f}  acc={accuracy_score(y, p > 0.5):.3f}"
          f"  | hard-only AUC={roc_auc_score(y[hard], p[hard]):.3f}"
          f"  acc={accuracy_score(y[hard], p[hard] > 0.5):.3f}")

print("\n=== TEST RESULTS ===")
report("Model1 predictions -> Model2", "pred")
report("Ground-truth attributes -> Model2", "gt")