import os
import torch
import clip
import numpy as np
from torch.utils.data import DataLoader

# Assume these functions/classes are defined elsewhere in your project.
# They are used to load the CelebA dataset and prepare train/valid/test splits.
from dataloader import ImageLoader, ImageDataset

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pretrained CLIP model and its preprocessing transform
model, clip_preprocess = clip.load("ViT-B/32", device=device)


# Define a helper function for computing fairness metrics (EOD and DP)
def get_fairness_metrics(preds, labels, sens):
    """
    calculate EOD, DP
    preds, labels, sens : 1dim, range 0/1
    return: eod, dp
    """
    preds = preds.view(-1).cpu()
    labels = labels.view(-1).cpu()
    sens = sens.view(-1).cpu()

    # indices for sensitive groups
    idx_s0 = (sens == 0)
    idx_s1 = (sens == 1)

    # DP: compute the positive prediction rates for each sensitive group
    p_y1_s0 = preds[idx_s0].float().mean() if idx_s0.sum() > 0 else 0.0
    p_y1_s1 = preds[idx_s1].float().mean() if idx_s1.sum() > 0 else 0.0
    dp = abs(p_y1_s0 - p_y1_s1)

    # EOD: compute TPR divergence = |TPR(s=0) - TPR(s=1)|
    # TPR = #(pred=1 & label=1) / (#(label=1))
    idx_s0_y1 = idx_s0 & (labels == 1)
    idx_s1_y1 = idx_s1 & (labels == 1)
    tpr_s0 = (preds[idx_s0_y1] == 1).sum() / (idx_s0_y1.sum() + 1e-7) if idx_s0_y1.sum() > 0 else 0.0
    tpr_s1 = (preds[idx_s1_y1] == 1).sum() / (idx_s1_y1.sum() + 1e-7) if idx_s1_y1.sum() > 0 else 0.0
    eod = abs(tpr_s0 - tpr_s1)

    return eod.item(), dp.item()


# Define arguments for dataset loading and prediction
class Args:
    dataname = 'celeba'
    bs_test = 64  # Batch size for testing
    label = 'Smiling'  # Target attribute for prediction


args = Args()

# Load CelebA dataset splits using a custom ImageLoader function
train_set, valid_set, test_set = ImageLoader(dataname=args.dataname)

# For CelebA, create the test dataset for the 'Smiling' attribute.
# It is assumed that the dataset returns the sensitive attribute "Male" as the second output.
if args.dataname == 'celeba':
    test_data = ImageDataset(test_set, args.dataname, 'Smiling', args.label,
                             './data/CelebA/img_align_celeba', clip_preprocess)
else:
    raise ValueError("Unsupported dataset name")

# Create a DataLoader for the test set
testloader = DataLoader(test_data, batch_size=args.bs_test, shuffle=False,
                        num_workers=8, pin_memory=True)

# Define candidate text prompts for zero-shot classification.
# Mapping: index 0 -> "not smiling" (label 0), index 1 -> "smiling" (label 1)
text_prompts = [
    "a photo of a person not smiling",  # corresponds to label 0
    "a photo of a person smiling"  # corresponds to label 1
]

# Tokenize the text prompts and compute their features once
text_inputs = torch.cat([clip.tokenize(prompt) for prompt in text_prompts]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Initialize accumulators for overall accuracy and fairness metric computation
total = 0
correct = 0
all_preds = []
all_labels = []
all_sens = []  # Sensitive attribute: Male

model.eval()
with torch.no_grad():
    for images, sens, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        sens = sens.to(device)  # "Male" sensitive attribute

        # Ensure labels are integer type for comparison
        if labels.dtype != torch.long:
            labels = labels.long()

        # Compute image features and normalize them
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity scores and convert to probabilities via softmax
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Get predictions: index 0 corresponds to label 0 ("not smiling"), index 1 to label 1 ("smiling")
        predictions = similarity.argmax(dim=1)

        correct += (predictions == labels.view(-1)).sum().item()
        total += labels.size(0)

        # Accumulate predictions, ground truth labels, and sensitive attributes
        all_preds.append(predictions.cpu())
        all_labels.append(labels.cpu())
        all_sens.append(sens.cpu())

# Calculate overall test accuracy
accuracy = correct / total * 100
print("Test Accuracy on CelebA (Smiling attribute): {:.2f}%".format(accuracy))

# Concatenate all predictions, labels, and sensitive attributes into single tensors
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
all_sens = torch.cat(all_sens)

# Compute fairness metrics using the helper function
eod, dp = get_fairness_metrics(all_preds, all_labels, all_sens)

# Output the fairness metrics
print("\nFairness Metrics:")
print("Equality of Opportunity (EOD) difference: {:.4f}".format(eod))
print("Demographic Parity (DP) difference: {:.4f}".format(dp))
