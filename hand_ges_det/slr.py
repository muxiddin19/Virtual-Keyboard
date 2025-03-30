import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

# Configure GPU memory growth to avoid memory allocation errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 1. Feature Extraction
class HandFeatureExtractor:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_landmarks(self, frame):
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.hands.process(frame_rgb)
        
        landmarks_list = []
        handedness_list = []
        
        # Check if hands were detected
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand type (left/right)
                handedness = results.multi_handedness[idx].classification[0].label
                
                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                landmarks_list.append(landmarks)
                handedness_list.append(handedness)
                
                # Draw landmarks on frame for visualization
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return frame, landmarks_list, handedness_list
    
    def close(self):
        self.hands.close()

# 2. Dataset Class
class SignLanguageDataset:
    def __init__(self, sequence_length=30, overlap=0.5):
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.extractor = HandFeatureExtractor()
        self.classes = []
        self.X_train = []
        self.y_train = []
        self.X_val = []
        self.y_val = []
        
    def process_video(self, video_path, label):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract hand landmarks
            _, landmarks_list, _ = self.extractor.extract_landmarks(frame)
            
            # Use the first detected hand's landmarks or zero padding if no hand detected
            if landmarks_list:
                frames.append(landmarks_list[0])  # Use first hand's landmarks
            else:
                frames.append([0.0] * 63)  # Zero padding for 21 landmarks (x,y,z)
        
        cap.release()
        
        # Create sequences with overlap
        sequences = []
        step = int(self.sequence_length * (1 - self.overlap))
        
        for i in range(0, len(frames) - self.sequence_length + 1, step):
            sequences.append(frames[i:i + self.sequence_length])
        
        return sequences, [label] * len(sequences)
    
    def load_dataset(self, dataset_dir, train_split=0.8):
        self.classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        class_to_index = {cls: i for i, cls in enumerate(self.classes)}
        
        all_sequences = []
        all_labels = []
        
        # Process each video for each class
        for class_name in self.classes:
            class_dir = os.path.join(dataset_dir, class_name)
            video_files = [f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi'))]
            
            for video_file in video_files:
                video_path = os.path.join(class_dir, video_file)
                sequences, labels = self.process_video(video_path, class_to_index[class_name])
                all_sequences.extend(sequences)
                all_labels.extend(labels)
        
        # Convert to numpy arrays
        all_sequences = np.array(all_sequences)
        all_labels = np.array(all_labels)
        
        # Shuffle the dataset
        indices = np.random.permutation(len(all_sequences))
        all_sequences = all_sequences[indices]
        all_labels = all_labels[indices]
        
        # Split into train and validation
        split_idx = int(len(all_sequences) * train_split)
        self.X_train = all_sequences[:split_idx]
        self.y_train = all_labels[:split_idx]
        self.X_val = all_sequences[split_idx:]
        self.y_val = all_labels[split_idx:]
        
        print(f"Loaded {len(self.X_train)} training sequences and {len(self.X_val)} validation sequences")
        print(f"Classes: {self.classes}")
        
        return self.X_train, self.y_train, self.X_val, self.y_val, self.classes
    
    def save_processed_dataset(self, output_path):
        """Save processed dataset to disk"""
        with open(output_path, 'wb') as f:
            pickle.dump({
                'X_train': self.X_train,
                'y_train': self.y_train,
                'X_val': self.X_val,
                'y_val': self.y_val,
                'classes': self.classes
            }, f)
        
    def load_processed_dataset(self, input_path):
        """Load processed dataset from disk"""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            self.X_train = data['X_train']
            self.y_train = data['y_train']
            self.X_val = data['X_val']
            self.y_val = data['y_val']
            self.classes = data['classes']
        
        return self.X_train, self.y_train, self.X_val, self.y_val, self.classes

# 3. Data Augmentation
class HandLandmarkAugmenter:
    """Data augmentation for hand landmarks"""
    
    @staticmethod
    def add_noise(landmarks, noise_level=0.01):
        """Add random noise to landmarks"""
        noise = np.random.normal(0, noise_level, landmarks.shape)
        return landmarks + noise
    
    @staticmethod
    def scale(landmarks, scale_range=(0.9, 1.1)):
        """Scale landmarks"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        # Only scale x,y coordinates (every 3rd element)
        scaled = landmarks.copy()
        scaled[:, :, 0::3] *= scale  # x coordinates
        scaled[:, :, 1::3] *= scale  # y coordinates
        return scaled
    
    @staticmethod
    def rotate(landmarks, angle_range=(-15, 15)):
        """Rotate landmarks in 2D space"""
        angle = np.random.uniform(angle_range[0], angle_range[1])
        angle_rad = np.radians(angle)
        
        rotated = landmarks.copy()
        
        # Extract x and y coordinates
        x = landmarks[:, :, 0::3]  # x coordinates
        y = landmarks[:, :, 1::3]  # y coordinates
        
        # Apply rotation
        rotated_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        rotated_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        # Put back to original structure
        rotated[:, :, 0::3] = rotated_x
        rotated[:, :, 1::3] = rotated_y
        
        return rotated
    
    @staticmethod
    def time_warp(landmarks, warp_factor=0.2):
        """Warp time dimension slightly (stretch/compress)"""
        seq_len = landmarks.shape[1]
        warp = np.random.uniform(1-warp_factor, 1+warp_factor)
        
        # Create new time indices
        old_indices = np.arange(seq_len)
        new_indices = np.linspace(0, seq_len-1, seq_len) * warp
        new_indices = np.clip(new_indices, 0, seq_len-1)
        
        # Interpolate
        warped = np.zeros_like(landmarks)
        for i in range(landmarks.shape[0]):
            for j in range(landmarks.shape[2]):
                warped[i, :, j] = np.interp(new_indices, old_indices, landmarks[i, :, j])
        
        return warped
    
    @staticmethod
    def augment_batch(X_batch, augmentation_probability=0.5):
        """Apply random augmentations to a batch"""
        X_aug = X_batch.copy()
        
        for i in range(len(X_aug)):
            if np.random.random() < augmentation_probability:
                # Choose random augmentations
                if np.random.random() < 0.5:
                    X_aug[i] = HandLandmarkAugmenter.add_noise(X_aug[i:i+1])[0]
                if np.random.random() < 0.5:
                    X_aug[i] = HandLandmarkAugmenter.scale(X_aug[i:i+1])[0]
                if np.random.random() < 0.5:
                    X_aug[i] = HandLandmarkAugmenter.rotate(X_aug[i:i+1])[0]
                if np.random.random() < 0.5:
                    X_aug[i] = HandLandmarkAugmenter.time_warp(X_aug[i:i+1])[0]
        
        return X_aug

# 4. Custom Data Generator
class SignLanguageDataGenerator(tf.keras.utils.Sequence):
    """Data generator for training with on-the-fly augmentation"""
    
    def __init__(self, X, y, batch_size=32, shuffle=True, augment=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.X))
        self.augmenter = HandLandmarkAugmenter()
        self.on_epoch_end()
    
    def __len__(self):
        """Return the number of batches per epoch"""
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indices of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        # Apply augmentation
        if self.augment:
            X_batch = self.augmenter.augment_batch(X_batch)
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        """Updates indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

# 5. Advanced Model Architecture
def build_advanced_sign_language_model(input_shape, num_classes):
    """
    Build an advanced sign language recognition model with:
    - Temporal convolutional layers
    - Bidirectional LSTM layers
    - Attention mechanism
    - Residual connections
    """
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # Normalization layer
    norm_layer = tf.keras.layers.LayerNormalization()(inputs)
    
    # 1D Convolutional layers for temporal feature extraction
    conv1 = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(norm_layer)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.SpatialDropout1D(0.1)(conv1)
    
    conv2 = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.SpatialDropout1D(0.1)(conv2)
    
    # Residual connection
    res_conn = tf.keras.layers.Add()([conv1, tf.keras.layers.Conv1D(128, kernel_size=1, padding='same')(conv1)])
    combined = tf.keras.layers.Add()([conv2, res_conn])
    
    # Bidirectional LSTM layers for sequence modeling
    lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(combined)
    lstm1 = tf.keras.layers.BatchNormalization()(lstm1)
    
    lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(lstm1)
    lstm2 = tf.keras.layers.BatchNormalization()(lstm2)
    
    # Self-attention mechanism
    attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(lstm2, lstm2)
    attention = tf.keras.layers.LayerNormalization()(attention + lstm2)  # Skip connection
    
    # Global temporal pooling
    gap = tf.keras.layers.GlobalAveragePooling1D()(attention)
    
    # Final dense layers
    dense1 = tf.keras.layers.Dense(256, activation='relu')(gap)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1 = tf.keras.layers.Dropout(0.3)(dense1)
    
    dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    dense2 = tf.keras.layers.Dropout(0.2)(dense2)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(dense2)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# 6. Training
def train_model(X_train, y_train, X_val, y_val, num_classes, sequence_length, num_features):
    """Train the model with advanced techniques"""
    
    # Define input shape
    input_shape = (sequence_length, num_features)
    
    # Build model
    model = build_advanced_sign_language_model(input_shape, num_classes)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    callbacks = [
        # Learning rate scheduler
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_sign_language_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

class HandLandmarkAugmenter:
    """Data augmentation for hand landmarks"""
    
    @staticmethod
    def add_noise(landmarks, noise_level=0.01):
        """Add random noise to landmarks"""
        noise = np.random.normal(0, noise_level, landmarks.shape)
        return landmarks + noise
    
    @staticmethod
    def scale(landmarks, scale_range=(0.9, 1.1)):
        """Scale landmarks"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        # Only scale x,y coordinates (every 3rd element)
        scaled = landmarks.copy()
        scaled[:, :, 0::3] *= scale  # x coordinates
        scaled[:, :, 1::3] *= scale  # y coordinates
        return scaled
    
    @staticmethod
    def rotate(landmarks, angle_range=(-15, 15)):
        """Rotate landmarks in 2D space"""
        angle = np.random.uniform(angle_range[0], angle_range[1])
        angle_rad = np.radians(angle)
        
        rotated = landmarks.copy()
        
        # Extract x and y coordinates
        x = landmarks[:, :, 0::3]  # x coordinates
        y = landmarks[:, :, 1::3]  # y coordinates
        
        # Apply rotation
        rotated_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        rotated_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        # Put back to original structure
        rotated[:, :, 0::3] = rotated_x
        rotated[:, :, 1::3] = rotated_y
        
        return rotated
    
    @staticmethod
    def time_warp(landmarks, warp_factor=0.2):
        """Warp time dimension slightly (stretch/compress)"""
        seq_len = landmarks.shape[1]
        warp = np.random.uniform(1-warp_factor, 1+warp_factor)
        
        # Create new time indices
        old_indices = np.arange(seq_len)
        new_indices = np.linspace(0, seq_len-1, seq_len) * warp
        new_indices = np.clip(new_indices, 0, seq_len-1)
        
        # Interpolate
        warped = np.zeros_like(landmarks)
        for i in range(landmarks.shape[0]):
            for j in range(landmarks.shape[2]):
                warped[i, :, j] = np.interp(new_indices, old_indices, landmarks[i, :, j])
        
        return warped
    
    @staticmethod
    def augment_batch(X_batch, augmentation_probability=0.5):
        """Apply random augmentations to a batch"""
        X_aug = X_batch.copy()
        
        for i in range(len(X_aug)):
            if np.random.random() < augmentation_probability:
                # Choose random augmentations
                if np.random.random() < 0.5:
                    X_aug[i] = HandLandmarkAugmenter.add_noise(X_aug[i:i+1])[0]
                if np.random.random() < 0.5:
                    X_aug[i] = HandLandmarkAugmenter.scale(X_aug[i:i+1])[0]
                if np.random.random() < 0.5:
                    X_aug[i] = HandLandmarkAugmenter.rotate(X_aug[i:i+1])[0]
                if np.random.random() < 0.5:
                    X_aug[i] = HandLandmarkAugmenter.time_warp(X_aug[i:i+1])[0]
        
        return X_aug
    
    class SignLanguageDataGenerator(tf.keras.utils.Sequence):
        """Data generator for training with on-the-fly augmentation"""
        
        def __init__(self, X, y, batch_size=32, shuffle=True, augment=True):
            self.X = X
            self.y = y
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = augment
            self.indices = np.arange(len(self.X))
            self.augmenter = HandLandmarkAugmenter()
            self.on_epoch_end()
        
        def __len__(self):
            """Return the number of batches per epoch"""
            return int(np.ceil(len(self.X) / self.batch_size))
        
        def __getitem__(self, index):
            """Generate one batch of data"""
            # Generate indices of the batch
            batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
            
            # Generate data
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]
            
            # Apply augmentation
            if self.augment:
                X_batch = self.augmenter.augment_batch(X_batch)
            
            return X_batch, y_batch
        
        def on_epoch_end(self):
            """Updates indices after each epoch"""
            if self.shuffle:
                np.random.shuffle(self.indices)