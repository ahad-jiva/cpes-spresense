import os
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import random

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4  # ['left', 'right', 'middle', 'none']
VALIDATION_SPLIT = 0.1  # fraction of labeled data to use for validation

# Paths
train_dir = 'widerface/train/images'
train_labels_file = 'widerface/train/label.txt'

# Label mapping
label_names = ['left', 'right', 'middle', 'none']
label_to_index = {name: idx for idx, name in enumerate(label_names)}

# making sure tf is using the gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            print(gpu)
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow is using the GPU")
    except RuntimeError as e:
        print(e)
else:
    print("TensorFlow is not using the GPU")

# load filenames and labels, label image according to bounds
def load_labels(labels_file, images_dir):
    file_paths = []
    labels = []
    with open(labels_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith('#'):
            rel_path = lines[i][1:].strip()
            img_path = os.path.join(images_dir, rel_path)
            i += 1
            parts = lines[i].split()
            label_name = 'none'
            if len(parts) >= 4:
                x, y, w, h = map(float, parts[:4])
                with Image.open(img_path) as img:
                    width, _ = img.size
                center_x = x + w / 2.0
                rel_center = center_x / width
                if rel_center < 1/3:
                    label_name = 'left'
                elif rel_center > 2/3:
                    label_name = 'right'
                else:
                    label_name = 'middle'
            file_paths.append(img_path)
            labels.append(label_to_index[label_name])
        i += 1
    return file_paths, labels

def preprocess_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image, tf.one_hot(label, NUM_CLASSES)

# splitting data for training and validation
def prepare_train_val_datasets(labels_file, images_dir, shuffle=True, val_split=VALIDATION_SPLIT):
    file_paths, labels = load_labels(labels_file, images_dir)
    data = list(zip(file_paths, labels))
    if shuffle:
        random.shuffle(data)
    num_val = int(len(data) * val_split)
    val_data = data[:num_val]
    train_data = data[num_val:]

    def make_ds(data_list, shuffle_ds=False):
        paths, labs = zip(*data_list)
        ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labs)))
        ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle_ds:
            ds = ds.shuffle(buffer_size=len(paths))
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(train_data, shuffle_ds=True)
    val_ds = make_ds(val_data)
    return train_ds, val_ds

train_ds, val_ds = prepare_train_val_datasets(train_labels_file, train_dir)

# randomly augment data (flip, rotate, zoom)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# model architecture
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = models.Model(inputs, outputs)

# compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# train
initial_epochs = 10
callbacks = [
  tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
  tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs,
    callbacks = callbacks
)

# Final evaluation on validation set
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation accuracy: {val_acc:.4f}")

model.export('mymodel')
train_ds.save('train_data')
print("Model and training data saved to disk.")



