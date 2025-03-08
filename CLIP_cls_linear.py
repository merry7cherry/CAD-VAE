import os
import torch
import clip
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import custom dataset loader functions.
# It is assumed that ImageLoader returns train/validation/test splits and
# ImageDataset returns tuples: (image, sensitive attribute, label)
from dataloader import ImageLoader, ImageDataset

# Set device to GPU if available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pretrained CLIP model and its preprocessing transform.
model, clip_preprocess = clip.load('ViT-B/32', device=device)


# ------------------------------------------------------------------
# Function to extract image features from a dataset using CLIP.
# The dataset is expected to return (image, sensitive attribute, label)
# ------------------------------------------------------------------
def get_features(dataset, batch_size=64):
    all_features = []
    all_labels = []
    all_sens = []
    # Create a DataLoader for the dataset.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)
    # Disable gradient computation for feature extraction.
    with torch.no_grad():
        for images, sens, labels in tqdm(loader):
            # Move images to device and extract CLIP features.
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)
            all_sens.append(sens)
    # Concatenate all batches and move to CPU as numpy arrays.
    return (torch.cat(all_features).cpu().numpy(),
            torch.cat(all_labels).cpu().numpy(),
            torch.cat(all_sens).cpu().numpy())


# ------------------------------------------------------------------
# Function to compute fairness metrics (Equality of Opportunity and Demographic Parity)
# Inputs:
#   preds, labels, sens: 1D torch Tensors with values 0/1.
# Returns:
#   eod: Absolute difference of True Positive Rates between sensitive groups.
#   dp: Absolute difference in positive prediction rates between sensitive groups.
# ------------------------------------------------------------------
def get_fairness_metrics(preds, labels, sens):
    preds = preds.view(-1).cpu()
    labels = labels.view(-1).cpu()
    sens = sens.view(-1).cpu()

    # Identify indices for the two sensitive groups (assumed to be 0 and 1)
    idx_s0 = (sens == 0)
    idx_s1 = (sens == 1)

    # Demographic Parity (DP): compute the difference in positive prediction rates.
    p_y1_s0 = preds[idx_s0].float().mean() if idx_s0.sum() > 0 else 0.0
    p_y1_s1 = preds[idx_s1].float().mean() if idx_s1.sum() > 0 else 0.0
    dp = abs(p_y1_s0 - p_y1_s1)

    # Equality of Opportunity Difference (EOD):
    # Compute True Positive Rates (TPR) for each sensitive group.
    idx_s0_y1 = idx_s0 & (labels == 1)
    idx_s1_y1 = idx_s1 & (labels == 1)
    tpr_s0 = (preds[idx_s0_y1] == 1).sum() / (idx_s0_y1.sum() + 1e-7) if idx_s0_y1.sum() > 0 else 0.0
    tpr_s1 = (preds[idx_s1_y1] == 1).sum() / (idx_s1_y1.sum() + 1e-7) if idx_s1_y1.sum() > 0 else 0.0
    eod = abs(tpr_s0 - tpr_s1)

    return eod.item(), dp.item()


# ------------------------------------------------------------------
# Define arguments for dataset and prediction task.
# ------------------------------------------------------------------
class Args:
    dataname = 'celeba'
    bs = 64  # Batch size for both training and testing.
    label = 'Smiling'  # Target attribute for prediction.


args = Args()

# Load CelebA dataset splits using the custom ImageLoader.
train_set, valid_set, test_set = ImageLoader(dataname=args.dataname)

# Create training and test datasets for the 'Smiling' attribute.
# It is assumed that the dataset returns the sensitive attribute "Male" as the second output.
if args.dataname == 'celeba':
    train_data = ImageDataset(train_set, args.dataname, 'Smiling', args.label,
                              './data/CelebA/img_align_celeba', clip_preprocess)
    test_data = ImageDataset(test_set, args.dataname, 'Smiling', args.label,
                             './data/CelebA/img_align_celeba', clip_preprocess)
else:
    raise ValueError("Unsupported dataset name")

# ------------------------------------------------------------------
# Extract features for training and testing using the CLIP image encoder.
# ------------------------------------------------------------------
print("Extracting features from training data...")
train_features, train_labels, train_sens = get_features(train_data, batch_size=args.bs)
print("Extracting features from test data...")
test_features, test_labels, test_sens = get_features(test_data, batch_size=args.bs)

# ------------------------------------------------------------------
# Train a logistic regression classifier (linear probe) using the extracted features.
# ------------------------------------------------------------------
print("Training logistic regression classifier...")
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# ------------------------------------------------------------------
# Evaluate the classifier on the test set.
# ------------------------------------------------------------------
print("Evaluating classifier on test data...")
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Test Accuracy on CelebA (Smiling attribute): {accuracy:.2f}%")

# Convert predictions, ground truth labels, and sensitive attributes to torch tensors
preds_tensor = torch.from_numpy(predictions)
labels_tensor = torch.from_numpy(test_labels)
sens_tensor = torch.from_numpy(test_sens)

# Compute fairness metrics using the helper function
eod, dp = get_fairness_metrics(preds_tensor, labels_tensor, sens_tensor)
print("\nFairness Metrics:")
print(f"Equality of Opportunity (EOD) difference: {eod:.4f}")
print(f"Demographic Parity (DP) difference: {dp:.4f}")
