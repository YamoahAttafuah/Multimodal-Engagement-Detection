import os
import cv2
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_curve, classification_report


# Defining data paths
PATH = r'C:\Users\yamoa\Downloads\DAiSEE\DAiSEE'
FEATURE_ROOT_DIR = r"C:\Users\yamoa\Downloads\DAiSEE\DAiSEE_Features_FF"

DATASET_PATH = os.path.join(PATH, 'DataSet')
LABELS_DIRECTORY_PATH = os.path.join(PATH, 'Labels')
ALL_LABELS_PATH = os.path.join(LABELS_DIRECTORY_PATH, 'AllLabels.csv')

SELECTED_FEATURES_PATH = 'daisee/selected_features_53.json'
MODEL_PATH = 'daisee/rf_daisee_model_GAMMA_53features.h5'   # For saving the model

# Defining constants
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 75   # No. of frames per clip
FEATURE_DIM = 1434    # 478 landmarks * 3 coordinates (x, y, z)

# Defining the weight penalty of each class for the loss function
CLASS_NAMES = ['Boredom', 'Confusion', 'Engagement', 'Frustration']
CLASS_WEIGHTS = [3.49, 10.60, 1.07, 20.73]
CLASS_WEIGHTS_TENSOR = tf.constant(CLASS_WEIGHTS, dtype=tf.float32)


# Loading indices
with open(SELECTED_FEATURES_PATH, 'r') as f:
    SURVIVOR_INDICES = json.load(f)


# Defining the data generator class
class DaiseeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, feature_root_dir, batch_size=32, max_seq_length=75, shuffle=True, oversample=False):
        self.df = df.copy().reset_index(drop=True)
        self.feature_root_dir = feature_root_dir
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.shuffle = shuffle

        self.map_of_paths = self._index_files()
        self.oversample = oversample
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))
    
    def _index_files(self):
        mapping = {}
        for root, directories, files in os.walk(self.feature_root_dir):
            for file in files:
                if file.endswith('npy'):
                    clip_id = os.path.basename(root)
                    mapping[clip_id] = os.path.join(root, file)
        return mapping

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size
        batch_indices = self.indices[start_index:end_index]
        
        num_features = len(SURVIVOR_INDICES)
        X = np.zeros((len(batch_indices), self.max_seq_length, num_features))
        y = np.zeros((len(batch_indices), 4))
        batch_df = self.df.iloc[batch_indices]

        for i, (_, row) in enumerate(batch_df.iterrows()):
            clip_id = str(row['ClipID']).split('.')[0]
            path = self.map_of_paths.get(clip_id)
            
            if path and os.path.exists(path):
                data = np.load(path)
                data_flat_full = data.reshape(data.shape[0], -1)
                data_reduced = data_flat_full[:, SURVIVOR_INDICES]
                
                length = min(len(data_reduced), self.max_seq_length)
                X[i, :length, :] = data_reduced[:length]
                
                y[i, 0] = row['Bored_binary']
                y[i, 1] = row['Confusion_binary']
                y[i, 2] = row['Engagement_binary']
                y[i, 3] = row['Frustration_binary']
        return X, y

    def get_resampled(self, indices, target_number):
        if len(indices) == 0: return []
        return np.random.choice(indices, size=target_number, replace=True) # Using replace=True to allow for picking the same sample multiple times

    def on_epoch_end(self):
        if self.oversample:
            # Identifying indices for each class based on the dataframe columns
            bored_idx = self.df[self.df['Bored_binary'] == 1].index.tolist()
            confused_idx = self.df[self.df['Confusion_binary'] == 1].index.tolist()
            engaged_idx = self.df[self.df['Engagement_binary'] == 1].index.tolist()
            frustrated_idx = self.df[self.df['Frustration_binary'] == 1].index.tolist()

            # Determining the size of the majority class
            class_counts = [len(bored_idx), len(confused_idx), len(engaged_idx), len(frustrated_idx)]
            max_count = max(class_counts) if any(class_counts) else 0

            if max_count > 0:
                # Resampling each class to match max_count
                b_oversampled = self.get_resampled(bored_idx, max_count)
                c_oversampled = self.get_resampled(confused_idx, max_count)
                e_oversampled = self.get_resampled(engaged_idx, max_count)
                f_oversampled = self.get_resampled(frustrated_idx, max_count)

                # Forming the new index list for this epoch
                self.indices = np.concatenate([b_oversampled, c_oversampled, e_oversampled, f_oversampled])
            else:
                self.indices = self.df.index.tolist()
        else:
            # Defining the regular non-oversampling approach for the test and validation sets
            self.indices = self.df.index.tolist()
            
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_and_preprocess_labels():
    # Retrieving labels
    all_labels_df = pd.read_csv(ALL_LABELS_PATH)

    # Defining the new mapping from 4 levels to 2 levels
    label_map = {0: 0, 1: 0, 2: 1, 3: 1} 
    all_labels_df['Bored_binary'] = all_labels_df['Boredom'].map(label_map)
    all_labels_df['Confusion_binary'] = all_labels_df['Confusion'].map(label_map)
    all_labels_df['Engagement_binary'] = all_labels_df['Engagement'].map(label_map)
    all_labels_df['Frustration_binary'] = all_labels_df['Frustration '].map(label_map)
    return all_labels_df


def create_subject_independent_splits(all_labels_df):
    # Extracting the subject ID
    all_labels_df['SubjectID'] = all_labels_df['ClipID'].astype(str).str[:6]

    # Finding the number of unique subjects
    unique_subjects = all_labels_df['SubjectID'].nunique()

    # Performing a subject-independent Split (ensuring all clips from a particular subject go into only the train set or only the test set)
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(all_labels_df, groups=all_labels_df['SubjectID']))

    intermediate_train_df = all_labels_df.iloc[train_idx]
    test_df = all_labels_df.iloc[test_idx]

    # Splitting the training set further to get a validation set
    val_splitter = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42) # 0.25 * 0.8 = 0.2
    train_final_idx, val_idx = next(val_splitter.split(intermediate_train_df, groups=intermediate_train_df['SubjectID']))

    train_df = intermediate_train_df.iloc[train_final_idx]
    val_df = intermediate_train_df.iloc[val_idx]

    print(f"\nFinal Split Counts:")
    print(f"Train: {len(train_df)} clips")
    print(f"Val:   {len(val_df)} clips")
    print(f"Test:  {len(test_df)} clips")
    return train_df, val_df, test_df


# Defining the Custom Loss Function (adapted from: https://pythonguides.com/binary-cross-entropy-tensorflow/)
def weighted_binary_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7) # to prevent log(0) errors

    # Loss = - pos weights* y*log(y) + (1-y)*log(1-y)
    loss = - (CLASS_WEIGHTS_TENSOR * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss)


# Defining the model architecture
def build_daisee_model(input_shape=(75, 54)):
    model = models.Sequential()
    model.add(layers.Input(input_shape))
    
    # LSTM layer
    model.add(layers.LSTM(64, return_sequences=False, dropout=0.3))
    
    # Dense layer
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    # Output layer
    model.add(layers.Dense(4, activation='sigmoid'))
    return model


# Defining the training callbacks and parameters
def train_model(model, train_gen, val_gen):
    # Defining callbacks
    callbacks = [
        # Save the best model automatically
        tf.keras.callbacks.ModelCheckpoint (
            filepath=MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        # Reduce the learning rate when performance does not improve
        tf.keras.callbacks.ReduceLROnPlateau (
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Early stopping to halt training if the model stops learning
        tf.keras.callbacks.EarlyStopping (
            monitor='val_loss',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    # Train
    history = model.fit (
        train_gen,
        validation_data=val_gen,
        epochs=40,
        callbacks=callbacks
    )
    return history


def evaluate_performance(model, test_gen, test_df, history):
    # Plot training history
    if history:
        metrics = history.history
        epochs_range = range(1, len(metrics['loss']) + 1)

        plt.figure(figsize=(18, 12))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, metrics['loss'], label='Training Loss', linewidth=2)
        plt.plot(epochs_range, metrics['val_loss'], label='Validation Loss', linestyle='--', linewidth=2)
        plt.title('Training vs Validation Loss', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, metrics['binary_accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(epochs_range, metrics['val_binary_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Training vs Validation Accuracy', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # F1-Score plot
        plt.subplot(2, 2, 4)
        
        # Calculate training F1
        p = np.array(metrics['precision'])
        r = np.array(metrics['recall'])
        f1 = 2 * (p * r) / (p + r + 1e-7)
        
        # Calculate validation F1
        val_p = np.array(metrics['val_precision'])
        val_r = np.array(metrics['val_recall'])
        val_f1 = 2 * (val_p * val_r) / (val_p + val_r + 1e-7)
        
        plt.plot(epochs_range, f1, label='Training F1-Score', linewidth=2)
        plt.plot(epochs_range, val_f1, label='Validation F1-Score', linestyle='--', linewidth=2)
        
        plt.title('Training vs Validation F1-Score', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('F1-Score', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        plt.tight_layout()
        
        # Saving the plot
        plt.savefig('training_history_large.png', dpi=300)
        print("Training history saved to 'training_history_large.png'")
        plt.show()

    # Thresholds and classification report
    print("Generating predictions")
    y_pred_probs = model.predict(test_gen, verbose=1)

    # Get Ground Truth Labels
    num_samples = len(test_gen) * test_gen.batch_size
    y_true = test_df.iloc[:num_samples][['Bored_binary', 'Confusion_binary', 'Engagement_binary', 'Frustration_binary']].values

    class_names = CLASS_NAMES
    best_thresholds = {}

    # Increased size for threshold plot
    plt.figure(figsize=(14, 8))

    for i, name in enumerate(class_names):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_thresholds[name] = best_thresh
        
        plt.plot(thresholds, f1_scores[:-1], label=f'{name} (Best: {best_thresh:.2f})', linewidth=2)
        print(f"Best Threshold for {name}: {best_thresh:.4f} -> Max F1: {best_f1:.4f}")

    plt.title("F1-Score vs Threshold", fontsize=16)
    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("F1-Score", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('threshold_optimization.png', dpi=300)
    plt.show()

    # Applying thresholds
    y_pred_optimized = np.zeros_like(y_pred_probs)
    for i, name in enumerate(class_names):
        thresh = best_thresholds[name]
        y_pred_optimized[:, i] = (y_pred_probs[:, i] >= thresh).astype(int)

    # Classification Report
    print("\nPerformance Report")
    print(classification_report(y_true, y_pred_optimized, target_names=class_names, zero_division=0))
    report_dict = classification_report(y_true, y_pred_optimized, target_names=class_names, zero_division=0, output_dict=True)
    return report_dict


def generate_accuracy_report(data):
    df = pd.DataFrame(data)

    # Computing total samples, True Positives, False Positives, True Negatives, False Negatives and Accuracy
    total_samples = df['Support'].sum()
    df['TP'] = (df['Recall'] * df['Support']).round().astype(int)
    df['FP'] = df.apply(lambda row: (row['TP'] / row['Precision'] - row['TP']) if row['Precision'] > 0 else 0, axis=1).round().astype(int)
    df['FN'] = df['Support'] - df['TP']
    df['TN'] = total_samples - (df['TP'] + df['FP'] + df['FN'])
    df['Accuracy'] = (df['TP'] + df['TN']) / total_samples

    # Displaying the results
    print("Accuracy Report")
    print(df[['Class', 'Accuracy', 'TP', 'FP', 'FN', 'TN']])

    # Weighted Average Accuracy
    print("\nSummary")
    weighted_acc = (df['Accuracy'] * df['Support']).sum() / total_samples
    print(f"Weighted Average Accuracy: {weighted_acc:.4f}")


def main():
    # Preparing data
    all_labels_df = load_and_preprocess_labels()
    train_df, val_df, test_df = create_subject_independent_splits(all_labels_df)
    
    # Instantiating generators
    train_gen = DaiseeDataGenerator(train_df, FEATURE_ROOT_DIR, batch_size=BATCH_SIZE, max_seq_length=MAX_SEQ_LENGTH, oversample=True)
    val_gen = DaiseeDataGenerator(val_df, FEATURE_ROOT_DIR, batch_size=BATCH_SIZE, max_seq_length=MAX_SEQ_LENGTH, shuffle=False)
    test_gen = DaiseeDataGenerator(test_df, FEATURE_ROOT_DIR, batch_size=BATCH_SIZE, max_seq_length=MAX_SEQ_LENGTH, shuffle=False)

    # Building model
    model = build_daisee_model()
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=weighted_binary_crossentropy, 
        metrics=['binary_accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    # Training
    history = train_model(model, train_gen, val_gen)

    # Evaluation
    report_dict = evaluate_performance(model, test_gen, test_df, history)

    # Accuracy Evaluation
    report_data = {
        'Class': CLASS_NAMES,
        'Precision': [report_dict[name]['precision'] for name in CLASS_NAMES],
        'Recall':    [report_dict[name]['recall']    for name in CLASS_NAMES],
        'Support':   [report_dict[name]['support']   for name in CLASS_NAMES]
    }
    generate_accuracy_report(report_data)


if __name__ == "__main__":
    main()