import torch
import torch.nn as nn
import torchvision.utils
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader
from dataloader import ImageDataset, ImageLoader, load_transform
from model import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
hdim = 512
feat_dim = 32
channel = [64, 128, 256, 512, 512]
img_size = 128
epoch = 180

# Directory paths
checkpoint_dir = "./save_iccv2025/celeba_Smiling/CAD-VAE_350.0_800.0_1.0_100.0_100.0_1.0_1.0_1.0_1.0_2"
img_save_dir = "./"

# Load the trained VAE model
encoder = nn.DataParallel(Encoder_CADVAE(hdim=hdim, feat_dim=feat_dim, channels=channel, image_size=img_size)).to(device)
decoder = nn.DataParallel(Decoder_Res(hdim=hdim, channels=channel, image_size=img_size)).to(device)

encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'encoder_{epoch}.pth'))['state_dict'])
decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'decoder_{epoch}.pth'))['state_dict'])

encoder.eval()
decoder.eval()

# Load the validation dataset
args = {
    'dataname': 'celeba',
    'label': 'Smiling',
    'bs': 64
}

train_set, valid_set, test_set = ImageLoader(dataname=args['dataname'])
transform, transform_test = load_transform(dataname=args['dataname'])

valid_data = ImageDataset(valid_set, args['dataname'], 'Male', args['label'], './data/CelebA/img_align_celeba',
                          transform_test)
validloader = DataLoader(valid_data, batch_size=10, shuffle=False, num_workers=8, pin_memory=True)

# Select sample images A and B
data_iter = iter(validloader)
x_valid, _, _ = next(data_iter)
x_valid = x_valid.to(device)

idx_A, idx_B = 1, 9  # Indices of images A and B in the batch
A = x_valid[idx_A].unsqueeze(0)
B = x_valid[idx_B].unsqueeze(0)

# Convert original images A and B for later display (values in [0,1])
img_A = A.squeeze().cpu().permute(1, 2, 0).numpy()
img_A = np.clip(img_A, 0, 1)
img_B = B.squeeze().cpu().permute(1, 2, 0).numpy()
img_B = np.clip(img_B, 0, 1)

# Generate latent codes for images A and B
with torch.no_grad():
    z_A, _, _ = encoder(A)
    z_B, _, _ = encoder(B)
    z_x_A, z_y_A, z_s_A, z_r_A = z_A
    z_x_B, z_y_B, z_s_B, z_r_B = z_B

# Linear interpolation coefficients
alpha_vals = torch.tensor([0.0, 0.4, 0.7, 1.1]).to(device)
zr_extra_vals = torch.tensor([0.5, 1.0]).to(device)

#####################################################
# Plot original 4×4 grid (varying zy and zs)
#####################################################
fig1, axes1 = plt.subplots(4, 4, figsize=(6.4, 6.4), dpi=300)
plt.subplots_adjust(wspace=0, hspace=0, left=0.1, right=0.9, top=0.9, bottom=0.1)

for i, alpha_y in enumerate(alpha_vals):
    for j, alpha_s in enumerate(alpha_vals):
        # Linear interpolation for zy and zs
        zy_interp = (1 - alpha_y) * z_y_A + alpha_y * z_y_B
        zs_interp = (1 - alpha_s) * z_s_A + alpha_s * z_s_B
        # Keep zr constant (original grid behavior)
        z_interp = torch.cat([z_x_A, zy_interp, zs_interp, z_r_A], dim=1)
        with torch.no_grad():
            recon = decoder(z_interp)
        img = recon.squeeze().cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes1[i, j].imshow(img)
        axes1[i, j].axis("off")

# Draw blue bounding box around the grid.
rect_blue = plt.Rectangle((0.10, 0.10), 0.80, 0.80, fill=False, edgecolor='blue', linewidth=1.5,
                           transform=fig1.transFigure, clip_on=False)
fig1.add_artist(rect_blue)

# Create an invisible axis for annotations.
ann_ax1 = fig1.add_axes([0, 0, 1, 1], zorder=10)
ann_ax1.axis('off')

# Add a downward arrow on the left of the blue box (for zy).
ann_ax1.annotate("", xy=(0.08, 0.12), xytext=(0.08, 0.88),
                 arrowprops=dict(arrowstyle="->", color='blue', lw=1.5),
                 xycoords='figure fraction', textcoords='figure fraction')
# Label the downward arrow ("zy") to its left.
ann_ax1.text(0.06, 0.50, f"$z_Y$", color='blue', fontsize=14, ha="right", va="center", transform=fig1.transFigure)

# Add a rightward arrow above the blue box (for zs).
ann_ax1.annotate("", xy=(0.88, 0.92), xytext=(0.12, 0.92),
                 arrowprops=dict(arrowstyle="->", color='blue', lw=1.5),
                 xycoords='figure fraction', textcoords='figure fraction')
# Label the rightward arrow ("zs") above it.
ann_ax1.text(0.50, 0.94, f"$z_S$", color='blue', fontsize=14, ha="center", va="bottom", transform=fig1.transFigure)

fig1.savefig(os.path.join(img_save_dir, f'interpolation_grid_original_{idx_A}_{idx_B}.png'),
             bbox_inches='tight', pad_inches=0)
plt.show()

#####################################################
# Plot additional 4×2 grid (varying zr)
#####################################################
fig2, axes2 = plt.subplots(4, 2, figsize=(3.2, 6.4), dpi=300)
plt.subplots_adjust(wspace=0, hspace=0, left=0.1, right=0.9, top=0.9, bottom=0.1)

for i, alpha_y in enumerate(alpha_vals):
    for k, alpha_r in enumerate(zr_extra_vals):
        # For the additional grid, keep zs fully interpolated to image B (zs = z_s_B)
        zy_interp = (1 - alpha_y) * z_y_A + alpha_y * z_y_B
        zs_interp = z_s_B
        zr_interp = (1 - alpha_r) * z_r_A + alpha_r * z_r_B
        z_interp = torch.cat([z_x_A, zy_interp, zs_interp, zr_interp], dim=1)
        with torch.no_grad():
            recon = decoder(z_interp)
        img = recon.squeeze().cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes2[i, k].imshow(img)
        axes2[i, k].axis("off")

# Draw red bounding box around the grid.
rect_red = plt.Rectangle((0.10, 0.10), 0.80, 0.80, fill=False, edgecolor='red', linewidth=1.5,
                          transform=fig2.transFigure, clip_on=False)
fig2.add_artist(rect_red)

ann_ax2 = fig2.add_axes([0, 0, 1, 1], zorder=10)
ann_ax2.axis('off')

# Add a rightward arrow above the red box (for zr).
ann_ax2.annotate("", xy=(0.88, 0.92), xytext=(0.12, 0.92),
                 arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
                 xycoords='figure fraction', textcoords='figure fraction')
# Label the arrow ("zr") above it.
ann_ax2.text(0.50, 0.94, f"$z_R$", color='red', fontsize=14, ha="center", va="bottom", transform=fig2.transFigure)

fig2.savefig(os.path.join(img_save_dir, f'interpolation_grid_additional_{idx_A}_{idx_B}.png'),
             bbox_inches='tight', pad_inches=0)
plt.show()

#####################################################
# Plot vertical image with original A and B
#####################################################
# Create a vertical figure with the same height (6.4 inches) and a canvas width of 2 inches.
fig3 = plt.figure(figsize=(2, 6.4), dpi=300)

# Add an axis for image A (top) with size equivalent to one grid cell (~1.6 inches square, i.e. 0.25 normalized height)
ax_A = fig3.add_axes([0.1, 0.60, 0.8, 0.25], zorder=11)
ax_A.imshow(img_A)
ax_A.axis("off")

# Add an axis for image B (bottom) with the same size.
ax_B = fig3.add_axes([0.1, 0.15, 0.8, 0.25], zorder=11)
ax_B.imshow(img_B)
ax_B.axis("off")

# Create an invisible axis for annotations in fig3.
ann_ax3 = fig3.add_axes([0, 0, 1, 1], zorder=10)
ann_ax3.axis('off')

# Add a black arrow from the bottom center of A to the top center of B.
ann_ax3.annotate("", xy=(0.5, 0.40), xytext=(0.5, 0.60),
                 arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                 xycoords='figure fraction', textcoords='figure fraction')

# Add text "Source" above image A.
fig3.text(0.5, 0.88, "Source", ha="center", va="bottom", fontsize=12, color="black")

# Add text "Reference" below image B.
fig3.text(0.5, 0.12, "Reference", ha="center", va="top", fontsize=12, color="black")

fig3.savefig(os.path.join(img_save_dir, f'interpolation_vertical_{idx_A}_{idx_B}.png'),
             bbox_inches='tight', pad_inches=0)
plt.show()
