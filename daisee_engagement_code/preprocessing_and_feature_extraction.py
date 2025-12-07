import os
import cv2
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from sklearn.model_selection import GroupShuffleSplit


def load_and_split_data(base_path):
    # Defining paths
    dataset_path = os.path.join(base_path, 'DataSet')
    labels_directory_path = os.path.join(base_path, 'Labels')
    all_labels = os.path.join(labels_directory_path, 'AllLabels.csv')
    all_labels_df = pd.read_csv(all_labels)

    # Defining the new mapping from 4 levels to 2 levels
    label_map = {0: 0, 1: 0, 2: 1, 3: 1} 
    all_labels_df['Bored_binary'] = all_labels_df['Boredom'].map(label_map)
    all_labels_df['Confusion_binary'] = all_labels_df['Confusion'].map(label_map)
    all_labels_df['Engagement_binary'] = all_labels_df['Engagement'].map(label_map)
    all_labels_df['Frustration_binary'] = all_labels_df['Frustration '].map(label_map)

    # Implementing subject-independence
    all_labels_df['SubjectID'] = all_labels_df['ClipID'].astype(str).str[:6]
    unique_subjects = all_labels_df['SubjectID'].nunique()

    # Performing a subject-independent Split (ensuring all clips from a particular subject go into only the train set or only the test set)
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(all_labels_df, groups=all_labels_df['SubjectID']))

    intermediate_train_df = all_labels_df.iloc[train_idx]
    test_df = all_labels_df.iloc[test_idx]

    # Splitting the intermediate training set further to get a validation set
    val_splitter = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42) # 0.25 * 0.8 = 0.2
    train_final_idx, val_idx = next(val_splitter.split(intermediate_train_df, groups=intermediate_train_df['SubjectID']))

    train_df = intermediate_train_df.iloc[train_final_idx]
    val_df = intermediate_train_df.iloc[val_idx]

    print(f"\nFinal Split Counts:")
    print(f"Train: {len(train_df)} clips")
    print(f"Val:   {len(val_df)} clips")
    print(f"Test:  {len(test_df)} clips")

    return train_df, val_df, test_df


# Extracting frames from clips
def get_frames_from_video(video_path, interval=4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
        
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % interval == 0:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        count += 1
    cap.release()


# Extracts 478 landmarks from a single frame
def extract_landmarks_from_frame(frame_rgb, face_mesh):
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        return coords
    else:
        return np.zeros((478, 3))


# Normalizes and scales
def normalize_landmarks_robust(landmarks):

    if np.all(landmarks == 0):
        return landmarks

    nose_tip_idx = 1
    left_eye_idx = 33   # Outer corner of left eye
    right_eye_idx = 263 # Outer corner of right eye

    nose = landmarks[nose_tip_idx]
    left_eye = landmarks[left_eye_idx]
    right_eye = landmarks[right_eye_idx]

    eye_dist = np.linalg.norm(left_eye - right_eye)

    if eye_dist < 0.0001:
        eye_dist = 1.0

    # Normalize and scale
    centered = landmarks - nose
    normalized = centered / eye_dist
    return normalized


# Runs the full extraction pipeline on one video.
def process_video_pipeline(video_path, face_mesh_detector):
    
    sequence_data = []
    
    for frame in get_frames_from_video(video_path, interval=4):
        raw_lms = extract_landmarks_from_frame(frame, face_mesh_detector)

        # If a face was found
        if not np.all(raw_lms == 0):
            norm_lms = normalize_landmarks_robust(raw_lms)
            sequence_data.append(norm_lms)
            
        # If no face was found and this is not the first frame
        elif len(sequence_data) > 0:
            # Forward fill with the previous valid frames
            sequence_data.append(sequence_data[-1])

        # If no face was found and this is the first frame, use zeros
        else:
            norm_lms = normalize_landmarks_robust(raw_lms)
            sequence_data.append(norm_lms)
    return np.array(sequence_data)


def process_entire_dataset(dataset_root, output_root, train_df, val_df, test_df):
    # Create a lookup dictionary
    clip_destination = {}
    for _, row in train_df.iterrows(): clip_destination[row['ClipID']] = 'Train'
    for _, row in val_df.iterrows():   clip_destination[row['ClipID']] = 'Validation'
    for _, row in test_df.iterrows():  clip_destination[row['ClipID']] = 'Test'

    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        # Iterate over the folders
        raw_source_splits = ['Train', 'Validation', 'Test']
        
        for source_split in raw_source_splits:
            split_path = os.path.join(dataset_root, source_split)
            if not os.path.exists(split_path):
                continue
            
            users = sorted(os.listdir(split_path))
            
            for user in users:
                user_path = os.path.join(split_path, user)
                if not os.path.isdir(user_path): continue

                extracts = os.listdir(user_path)
                
                # Progress bar
                for extract in tqdm(extracts, desc=f"Reading {source_split}/{user}", leave=False):
                    extract_path = os.path.join(user_path, extract)
                    if not os.path.isdir(extract_path): continue

                    # Find the video file
                    clip_files = [f for f in os.listdir(extract_path) if f.endswith('.avi')]
                    if len(clip_files) == 0: continue
                    
                    clip_name = clip_files[0]
                    video_full_path = os.path.join(extract_path, clip_name)

                    if clip_name not in clip_destination:
                        continue

                    target_split = clip_destination[clip_name]

                    # Save to the target split folder
                    save_dir = os.path.join(output_root, target_split, user, extract)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "features.npy")

                    if os.path.exists(save_path):
                        continue

                    # Process and save
                    try:
                        features = process_video_pipeline(video_full_path, face_mesh)
                        np.save(save_path, features)
                    except Exception as e:
                        print(f"Failed on {clip_name}: {e}")


def count_forward_filled_frames(features_root):
    total_frames = 0
    filled_frames = 0
    total_clips = 0
    clips_with_fills = 0

    # Get list of all .npy files
    npy_files = []
    for root, dirs, files in os.walk(features_root):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))

    # Iterate and count
    for file_path in tqdm(npy_files, desc="Analyzing clips"):
        try:
            # Load the feature array of shape (frames, 478, 3)
            data = np.load(file_path)
            
            if data.size == 0:
                continue

            # Count the total frames in this clip
            num_frames = data.shape[0]
            total_frames += num_frames
            total_clips += 1

            # Find the forward-filled frames
            if num_frames > 1:
                # Compare frames
                is_duplicate = np.all(data[1:] == data[:-1], axis=(1, 2))
                num_filled = np.sum(is_duplicate)
            else:
                num_filled = 0
            
            filled_frames += num_filled

            if num_filled > 0:
                clips_with_fills += 1

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if total_frames == 0:
        print("No data found.")
        return
    print("Forward-Fill Analysis Report")
    print(f"Total Clips Processed:   {total_clips}")
    print(f"Total Frames Processed:  {total_frames:,}")
    print(f"Frames Forward Filled:   {filled_frames:,}")
    print(f"Fill Rate:               {(filled_frames/total_frames)*100:.2f}%")
    print(f"Clips containing fills:  {clips_with_fills} ({(clips_with_fills/total_clips)*100:.2f}%)")


def analyze_feature_correlation(feature_root_dir, target_frame_count=30000):
    TARGET_FRAME_COUNT = target_frame_count

    # Find every .npy file
    all_feature_files = []
    for root, _, files in os.walk(feature_root_dir):
        for file in files:
            if file.endswith('.npy'):
                full_path = os.path.join(root, file)
                all_feature_files.append(full_path)

    # Shuffle the file list to ensure random selection
    random.seed(42)
    random.shuffle(all_feature_files)

    collected_frames = []
    total_frames = 0

    # Iterate through the shuffled list
    for file_path in all_feature_files:
        if total_frames >= TARGET_FRAME_COUNT:
            break
        
        try:
            # Load the raw data
            data = np.load(file_path)
            
            # Flatten from (frames, 478, 3) to (frames, 1434)
            if data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
                
            collected_frames.append(data)
            total_frames += data.shape[0]
            
        except Exception as e:
            print(f"Skipping file {file_path}: {e}")

    full_matrix = np.vstack(collected_frames)
    
    # Compute the correlation matrix
    df_features = pd.DataFrame(full_matrix)
    corr_matrix = df_features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with a correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    print(f"Found {len(to_drop)} redundant features (Corr > 0.95).")

    # Drop to see what remains
    df_reduced = df_features.drop(columns=to_drop)
    survivor_indices = df_reduced.columns.tolist()

    # Save the indices
    filename = f'selected_features_{len(survivor_indices)}.json'
    with open(filename, 'w') as f:
        json.dump(survivor_indices, f)

    print(f"Saved {len(survivor_indices)} indices to '{filename}'")
    print(f"Example indices: {survivor_indices[:10]}")


def calculate_class_weights(train_df):
    # Counting the positive samples for each class in the training set
    counts = []
    class_cols = ['Bored_binary', 'Confusion_binary', 'Engagement_binary', 'Frustration_binary']

    print("Class Distribution in Training Set:")
    total_samples = len(train_df)

    for col in class_cols:
        pos_count = train_df[col].sum()
        counts.append(pos_count)
        print(f"  {col}: {pos_count} / {total_samples} ({pos_count/total_samples:.2%})")

    # Calculating weights where weight = total samples / label samples
    pos_weights = [total_samples / c for c in counts]

    print(f"\nPositive Weights: {pos_weights}")
    return pos_weights


def main():
    # Defning paths
    MAIN_PATH = 'C:\\Users\\yamoa\\Downloads\\DAiSEE\\DAiSEE'
    DATASET_ROOT = 'C:\\Users\\yamoa\\Downloads\\DAiSEE\\DAiSEE\\DataSet'
    OUTPUT_ROOT = 'C:\\Users\\yamoa\\Downloads\\DAiSEE\\DAiSEE_Features_FF'
    FEATURES_DIR = OUTPUT_ROOT
    
    # Process labels and split
    train_df, val_df, test_df = load_and_split_data(MAIN_PATH)

    # Process dataset
    process_entire_dataset(DATASET_ROOT, OUTPUT_ROOT, train_df, val_df, test_df)

    # Analyse how many frames required forward-filling
    count_forward_filled_frames(FEATURES_DIR)

    # Perform feature correlation and reduction
    analyze_feature_correlation(FEATURES_DIR)

    # Calculate class weights
    calculate_class_weights(train_df)


if __name__ == "__main__":
    main()