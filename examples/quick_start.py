"""
Quick Start Example for NILM

Minimal example to get started with the NILM package.
"""

import os
import sys
import numpy as np

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nilm.models import BiLSTM
from nilm.data import DataLoader, DataPreprocessor
from nilm.utils import MetricsCalculator


# Generate some sample data
data_loader = DataLoader()
X, y = data_loader.generate_synthetic_data(
    num_samples=1000,
    window_size=599,
    appliance="kettle"
)

# Split data
preprocessor = DataPreprocessor(window_size=599)
X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)

# Create and train model
model = BiLSTM(
    appliance_name="kettle",
    window_size=599,
    lstm_units=[64, 32]
)

print("Training BiLSTM model...")
model.train(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# Evaluate
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, None)

metrics = MetricsCalculator("kettle")
results = metrics.calculate_all(y_test, y_pred)
metrics.print_metrics(results)

print("Done!")
