import tensorflow as tf, pandas as pd, json
from tensorflow.keras import layers, Model

IMG_SIZE, BATCH, EPOCHS = 128, 64, 10
HEADS = ["article", "color", "usage", "season"]
AUTOTUNE = tf.data.AUTOTUNE

label_maps = json.load(open("work/label_maps.json"))
n_classes = {h: len(label_maps[h]) for h in HEADS}

def make_ds(csv_path, training):
    df = pd.read_csv(csv_path)
    paths = df["path"].values
    labels = {h: df[h].values.astype("int32") for h in HEADS}
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(p, y):
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))   # upscale 60x80 -> 128x128
        return img, y

    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(2000, seed=42)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

train_ds = make_ds("work/train.csv", True)
val_ds   = make_ds("work/val.csv", False)

# --- model ---
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.1),
    layers.RandomRotation(0.05),
])  # no colour jitter on purpose: it would hurt the colour head

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False

inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
x = augment(inp)
x = layers.Rescaling(1/127.5, offset=-1)(x)      # MobileNetV2 expects [-1, 1]
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = {h: layers.Dense(n_classes[h], activation="softmax", name=h)(x) for h in HEADS}
model1 = Model(inp, outputs)

model1.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss={h: "sparse_categorical_crossentropy" for h in HEADS},
    metrics={h: "accuracy" for h in HEADS},
)
model1.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("work/model1_stage1.keras", save_best_only=True,
                                       monitor="val_loss"),
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                     monitor="val_loss"),
]
model1.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# --- per-head validation accuracy ---
res = model1.evaluate(val_ds, return_dict=True)
for h in HEADS:
    print(f"{h:8s} val accuracy: {res[f'{h}_accuracy']:.3f}")