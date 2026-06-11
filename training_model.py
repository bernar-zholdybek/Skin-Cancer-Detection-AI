import os, pandas as pd, numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt, seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ==== Пути ====
BASE = "HAM10000"
META_PATH = os.path.join(BASE, "HAM10000_metadata.csv")
PART1 = os.path.join(BASE, "HAM10000_images_part_1")
PART2 = os.path.join(BASE, "HAM10000_images_part_2")

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]
class_to_idx = {c:i for i,c in enumerate(CLASS_NAMES)}

# ==== Metadata uploading ====
meta = pd.read_csv(META_PATH)

def resolve_path(image_id):
    for part in [PART1, PART2]:
        p = os.path.join(part, f"{image_id}.jpg")
        if os.path.exists(p): return p
    return None

paths, labels = [], []
for _, row in meta[["image_id","dx"]].iterrows():
    p = resolve_path(row["image_id"])
    if p and row["dx"] in class_to_idx:
        paths.append(p)
        labels.append(class_to_idx[row["dx"]])

paths = np.array(paths)
labels = np.array(labels)
print("Number of images:", len(paths))

# ==== Class balancing ====
def balance_dataset(paths, labels, target_per_class=1000):
    new_paths, new_labels = [], []
    for i in range(len(CLASS_NAMES)):
        idxs = np.where(labels == i)[0]
        count = len(idxs)
        if count < target_per_class:
            extra = np.random.choice(idxs, size=target_per_class - count, replace=True)
            idxs = np.concatenate([idxs, extra])
        else:
            idxs = np.random.choice(idxs, size=target_per_class, replace=False)
        new_paths.extend(paths[idxs])
        new_labels.extend(labels[idxs])
    return np.array(new_paths), np.array(new_labels)

paths, labels = balance_dataset(paths, labels, target_per_class=1000)

# ==== Split: train/val/test ====
x_train, x_tmp, y_train, y_tmp = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
x_train, x_tmp, y_train, y_tmp = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.one_hot(label, depth=len(CLASS_NAMES))

data_augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

def augment(img, label):
    return data_augment(img, training=True), label

def make_ds(x, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training: ds = ds.shuffle(len(x))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training: ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_ds(x_train, y_train, training=True)
val_ds   = make_ds(x_val,   y_val)
test_ds  = make_ds(x_test,  y_test)

# ==== MobileNetV2 + Focal Loss ====
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # firstly frozing it

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(len(CLASS_NAMES), activation="softmax")
])


def focal_loss(gamma=2., alpha=0.25):
    """
    gamma - how much it focuses on hard examples
    alpha — balance between classes (usually 0.25 for rare classes)
    """

    def loss(y_true, y_pred):
        # standart crossentropy
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # probability of correct class
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        pt = tf.reduce_sum(y_true * y_pred_clipped, axis=-1)

        # focal modificator
        focal_factor = alpha * tf.pow((1. - pt), gamma)

        return focal_factor * ce

    return loss


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal_loss(gamma=2., alpha=0.25),
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

model.summary()

# ==== Training ====
history = model.fit(train_ds, epochs=30, validation_data=val_ds, callbacks=[early_stop])

# ==== Graphics ====
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Error!"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.tight_layout()
plt.show()

# ==== Evaluation on test ====
y_true, y_pred = [], []
for images, labels_onehot in test_ds:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels_onehot.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap="Blues")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ==== Model saving ====
model.save("ham10000_cnn_transfer_learning.h5")
print("Model is saved!")