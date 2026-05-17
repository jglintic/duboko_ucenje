import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow import keras
from tensorflow.keras.layers import TimeDistributed, LSTM
import matplotlib.pyplot as plt
import imageio.v2 as imageio

CSV_PATH = "driving_log.csv"
IMG_DIR = "IMG"
ORIGINAL_SIZE = (160, 320)
NEW_SIZE = (66, 200)
TEST_SIZE = 0.2

def load_image(img_path):
    img = load_img(img_path, target_size=ORIGINAL_SIZE)
    img = img_to_array(img)
    # CROP
    # original: 160x320
    # skidamo ~60px gore (nebo) i ~20px dole (hauba)
    img = img[60:140, :, :]   # sada je ~80x320

    # RESIZE na 66x200
    img = tf.image.resize(img, NEW_SIZE)

    # normalizacija
    img = img / 255.0

    return img.numpy()

print("Loading CSV file...")

df = pd.read_csv(CSV_PATH, header=None)
df.columns = [
    "center", "left", "right",
    "steering", "throttle", "reverse", "speed"
]

# koristimo samo centralnu kameru
df["center"] = df["center"].apply(
    lambda x: os.path.join(IMG_DIR, os.path.basename(x))
)

print(f"Ukupno uzoraka: {len(df)}")


# podela na train i test podatke
# rucni 80/20 split
split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

print("\nLoading training images...")

train_images = []
train_angles = []

for _, row in train_df.iterrows():
    img = load_image(row["center"])
    train_images.append(img)
    train_angles.append(row["steering"])

train_images = np.array(train_images)
train_angles = np.array(train_angles)

print("Train images loaded:")
print("Images shape:", train_images.shape)
print("Angles shape:", train_angles.shape)

print("\nLoading test images...")

test_images = []
test_angles = []

for _, row in test_df.iterrows():
    img = load_image(row["center"])
    test_images.append(img)
    test_angles.append(row["steering"])

test_images = np.array(test_images)
test_angles = np.array(test_angles)

print("Test images loaded:")
print("Images shape:", test_images.shape)
print("Angles shape:", test_angles.shape)

def create_sequences(images, angles, sequence_length=5):
    X = []
    y = []
    
    for i in range(len(images) - sequence_length + 1):
        
        seq_images = images[i:i+sequence_length]
        seq_angle = angles[i+sequence_length-1]
        
        X.append(seq_images)
        y.append(seq_angle)
    
    return np.array(X), np.array(y)

# kreiranje sekvenci od 5 slika
SEQUENCE_LENGTH = 5

print("\nCreating sequences...")

train_images_seq, train_angles_seq = create_sequences(
    train_images, train_angles, SEQUENCE_LENGTH
)

test_images_seq, test_angles_seq = create_sequences(
    test_images, test_angles, SEQUENCE_LENGTH
)

print("Train sequence shape:", train_images_seq.shape)
print("Train angles shape:", train_angles_seq.shape)

print("Test sequence shape:", test_images_seq.shape)
print("Test angles shape:", test_angles_seq.shape)

# CNN MODEL
print("\nBuilding CNN regression model...")

inputs = keras.Input(shape=(SEQUENCE_LENGTH, 66, 200, 3))

x = TimeDistributed(keras.layers.Conv2D(24, (5,5), activation="relu", strides=(2,2)))(inputs)
x = TimeDistributed(keras.layers.BatchNormalization())(x)
x = TimeDistributed(keras.layers.Conv2D(36, (5,5), activation="relu", strides=(2,2)))(x)
x = TimeDistributed(keras.layers.BatchNormalization())(x)
x = TimeDistributed(keras.layers.Conv2D(48, (5,5), activation="relu", strides=(2,2)))(x)
x = TimeDistributed(keras.layers.BatchNormalization())(x)
x = TimeDistributed(keras.layers.Conv2D(64, (3,3), activation="relu"))(x)
x = TimeDistributed(keras.layers.Conv2D(64, (3,3), activation="relu"))(x)

x = TimeDistributed(keras.layers.Flatten())(x)

# LSTM za sekvencijalne zavisnosti
x = LSTM(64, return_sequences=False)(x)
x = keras.layers.Dropout(0.4)(x)

x = keras.layers.Dense(32, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(10, activation="relu")(x)

# jedan izlaz - regresija
outputs = keras.layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    #optimizer="adam",
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="mse",      # regresija
    metrics=["mae"]
)

model.summary()

# sigurnosni mehanizam protiv overfitting-a
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\nStarting training...")

history = model.fit(
    train_images_seq,
    train_angles_seq,
    validation_data=(test_images_seq, test_angles_seq),
    epochs=15,
    batch_size=32,
    shuffle=True,
    callbacks=[early_stop]
)

print("\nEvaluating model...")
loss, mae = model.evaluate(test_images_seq, test_angles_seq)

print("Test MSE:", loss) # prosečna kvadratna greška na test podacima
print("Test MAE:", mae) # prosečna apsolutna greška

def predict_on_test_sequence(model, test_images_seq): 
    predictions = model.predict(test_images_seq, verbose=0)
    
    return predictions.flatten()

print("Generating predictions for test sequence...")

predicted_angles = predict_on_test_sequence(model, test_images_seq)

print("Number of predictions:", len(predicted_angles))

# iscrtavanje steering angle
def draw_steering_angle(image, angle):
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    h, w, _ = image.shape
    
    center_x = w // 2 # sredina slike
    center_y = h # dno slike
    
    length = 40 # duzina strelice
    
    end_x = center_x + length * np.sin(angle)
    end_y = center_y - length * np.cos(angle)
    
    ax.plot([center_x, end_x], [center_y, end_y], linewidth=3)
    
    ax.axis("off")
    
    fig.canvas.draw()
    
    result = np.asarray(fig.canvas.buffer_rgba())
    result = result[:, :, :3]
    
    plt.close(fig)
    
    return result

def add_lane_change_overlay(image):
    fig, ax = plt.subplots()
    ax.imshow(image)

    # crveni pravougaonik
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            image.shape[1],
            50,
            color='red',
            alpha=0.6
        )
    )

    ax.text(
        image.shape[1] // 2,
        30,
        "LANE CHANGE",
        color='white',
        fontsize=18,
        ha='center',
        va='center',
        weight='bold'
    )

    ax.axis("off")
    fig.canvas.draw()

    result = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    return result

def smooth_signal(signal, kernel_size=7):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(signal, kernel, mode='same')
    
def detect_lane_changes(predictions, center_threshold=0.08, lane_threshold=0.22, min_duration=15):
    """
    Smoothed predictions -> lane change signal.
    - min_duration: broj frejmova koliko obaveštenje traje
    """
    smoothed = smooth_signal(predictions, kernel_size=9)
    lane_changes = np.zeros(len(smoothed))

    state = "CENTER"
    active_counter = 0  # koliko frejmova signal traje

    for i, angle in enumerate(smoothed):
        if abs(angle) < center_threshold:
            state = "CENTER"
            active_counter = 0
        elif angle > lane_threshold:
            if state == "CENTER":
                active_counter = min_duration
            state = "LEFT"
        elif angle < -lane_threshold:
            if state == "CENTER":
                active_counter = min_duration
            state = "RIGHT"
        else:
            # ako ugao između centar i lane_threshold, zadržavamo prethodno stanje
            pass

        if active_counter > 0:
            lane_changes[i] = 1
            active_counter -= 1

    return lane_changes
    
def create_video_from_predictions(images, predictions, output_path="test_output.mp4"):
    writer = imageio.get_writer(output_path, format="FFMPEG", fps=20)
    
    lane_changes = detect_lane_changes(predictions)

    for i in range(len(predictions)):
        
        frame = images[i+4]
        frame_with_angle = draw_steering_angle(frame, predictions[i])

        if lane_changes[i] == 1:
            frame_with_angle = add_lane_change_overlay(frame_with_angle)

        writer.append_data(frame_with_angle.astype(np.uint8))

    writer.close()

create_video_from_predictions(test_images, predicted_angles, "test_output.mp4")