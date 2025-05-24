# train.py
# -*- coding: utf-8 -*-
#========================================================================================
#========================================================================================
# 出现过拟合，失败
#========================================================================================
#========================================================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# 只保留静态 24 类（跳过 J、Z）
CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','K',
    'L','M','N','O','P','Q','R','S','T','U',
    'V','W','X','Y'
]

# 数据增强函数，仅在训练时调用
def augment_images(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_zoom(image, (0.9,1.1)) if hasattr(tf.image, 'random_zoom') else image
    return image, label


def create_model(input_shape=(28,28,1)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2,2)(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2,2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def process_path(file_path, label, image_size=(28,28)):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, image_size)
    img = img / 255.0
    return img, label


def load_data(data_root, image_size=(28,28), batch_size=64, val_split=0.2, shuffle_buffer=1000):
    train_dir = os.path.join(data_root, 'asl_alphabet_train', 'asl_alphabet_train')
    all_paths, all_labels = [], []
    for idx, cls in enumerate(CLASS_NAMES):
        cls_folder = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_folder):
            continue
        for fname in os.listdir(cls_folder):
            if not fname.lower().endswith(('.jpg', '.png')):
                continue
            all_paths.append(os.path.join(cls_folder, fname))
            all_labels.append(idx)

    # 构建 dataset 并打乱
    dataset = tf.data.Dataset.from_tensor_slices((all_paths, all_labels))
    dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    total = len(all_paths)
    val_size = int(total * val_split)
    val_ds = dataset.take(val_size)
    train_ds = dataset.skip(val_size)

    def prepare_train(ds):
        ds = ds.map(lambda p, l: process_path(p, l, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(augment_images, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def prepare_val(ds):
        ds = ds.map(lambda p, l: process_path(p, l, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    return prepare_train(train_ds), prepare_val(val_ds)


if __name__ == "__main__":
    data_root = r"E:\360MoveData\Users\HYT\Desktop\人工智能交互实践\Project\data"
    train_ds, val_ds = load_data(data_root)

    model = create_model()

    checkpoint = ModelCheckpoint(
        'sign_language_cnn.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    history = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=[checkpoint]
    )

    # 保存训练与验证的准确率曲线
    plt.figure()
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')

    # 保存训练与验证的损失曲线
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {acc:.4f}")
