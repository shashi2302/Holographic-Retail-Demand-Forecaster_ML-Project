# Model training script
import tensorflow as tf
from data_loader import RetailDataLoader
from model import build_holographic_forecasting_model

# Initialize data loader
data_loader = RetailDataLoader(
    data_path="/path/to/processed/data",
    batch_size=32,
    sequence_length=24,  # 24 hours of data
    holo_dim=512
)

# Create train/validation/test splits
train_dataset, val_dataset, test_dataset = data_loader.create_datasets()

# Build model
input_shape = (24, 64, 64, 3)  # Time, height, width, features
model = build_holographic_forecasting_model(
    input_shape=input_shape,
    holo_dim=512,
    output_dim=6  # Predicting 6 demand metrics
)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Train model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir="./logs")
    ]
)

# Evaluate model
test_results = model.evaluate(test_dataset)
print(f"Test Loss: {test_results[0]}, Test MAE: {test_results[1]}")
