import tensorflow as tf, pandas as pd, json
from tensorflow.keras import layers

IMG_SIZE, BATCH, EPOCHS = 128, 64, 8
HEADS = ["article", "color", "usage", "season"]
AUTOTUNE = tf.data.AUTOTUNE

def make_ds(csv_path, training):
    df = pd.read_csv(csv_path)
    paths = df["path"].values
    labels = {h: df[h].values.astype("int32") for h in HEADS}
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def load(p, y):
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img, y
    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(2000, seed=42)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

train_ds = make_ds("work/train.csv", True)
val_ds   = make_ds("work/val.csv", False)

model1 = tf.keras.models.load_model("work/model1_stage1.keras")   # start again from stage 1

# find MobileNetV2 by name (not by type)
base = next(l for l in model1.layers if l.name.startswith("mobilenetv2"))
print("backbone:", base.name, "| total layers:", len(base.layers))

# unfreeze from layer 100 onward (about the last third), keep BatchNorm frozen
base.trainable = True
for l in base.layers[:100]:
    l.trainable = False
for l in base.layers[100:]:
    l.trainable = not isinstance(l, layers.BatchNormalization)
print("trainable backbone layers:", sum(l.trainable for l in base.layers))

model1.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={h: "sparse_categorical_crossentropy" for h in HEADS},
    metrics={h: "accuracy" for h in HEADS},
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("work/model1.keras", save_best_only=True,
                                       monitor="val_loss"),
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                     monitor="val_loss"),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1, monitor="val_loss"),
]
model1.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
           callbacks=callbacks, verbose=2)   # verbose=2 = one line per epoch

res = model1.evaluate(val_ds, return_dict=True, verbose=0)
for h in HEADS:
    print(f"{h:8s} val accuracy: {res[f'{h}_accuracy']:.3f}")