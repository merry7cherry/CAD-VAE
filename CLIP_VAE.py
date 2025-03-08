import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import argparse
import sys
import shutil

from dataloader import ImageLoader, ImageDataset, load_transform
from loss import *


parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=250)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--start_CMI_epoch", type=int, default=2)
parser.add_argument("--feat_dim", type=int, default=32)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--beta", type=float, default=1)
parser.add_argument("--lambda_cls", type=float, default=1e3)
parser.add_argument("--lambda_tc", type=float, default=1e0)
parser.add_argument("--lambda_fair_yzy", type=float, default=1e2)
parser.add_argument("--lambda_fair_szs", type=float, default=1e2)
parser.add_argument("--lambda_fair_yzs", type=float, default=1e0)
parser.add_argument("--lambda_fair_szy", type=float, default=1e0)
parser.add_argument("--log_name", type=str, default='')
parser.add_argument("--dataname", type=str, default='celeba')
parser.add_argument("--label", type=str, default='Smiling')
parser.add_argument("--resume", type=bool, default=False)
args = parser.parse_args()

args.log_name += 'zr_TC_Learn_CMI_FADES_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:d}'.format(
    args.bs,
    args.lambda_cls,
    args.lambda_tc,
    args.lambda_fair_yzy,
    args.lambda_fair_szs,
    args.lambda_fair_yzs,
    args.lambda_fair_szy,
    args.alpha,
    args.beta,
    args.start_CMI_epoch)
# based on IntroVAE
dataname = f'{args.dataname}'

if args.dataname == 'celeba':
    dataname += f'_{args.label}'

save_dir = os.path.join('./save_clip', dataname, args.log_name)

os.makedirs(save_dir, exist_ok=True)

# init log
with open(os.path.join(save_dir, "log.txt"), "a") as f:
    f.write(" ".join(sys.argv) + '\n')


def write_loss_to_log(logging):
    with open(os.path.join(save_dir, "log.txt"), "a") as f:
        f.write(logging)


# Get the script filename and save path
filename = sys.argv[0]
destination = os.path.join(save_dir, os.path.basename(filename))

# Only copy if source and destination are different
if filename != destination:
    shutil.copyfile(filename, destination)



# Assume the following functions and variables are defined elsewhere:
# direct_CMI_loss, learning_relevant_zr, permute_zs, write_loss_to_log
# args (with attributes: feat_dim, lambda_cls, lambda_tc, lambda_fair_yzs, lambda_fair_szy,
#       lambda_fair_yzy, lambda_fair_szs, alpha, beta, start_CMI_epoch, epochs, resume, save_dir, etc.)
# trainloader, validloader

#########################################
# Integrated VAE with built-in CLIP Feature Extraction
#########################################
class Project(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Project, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out)
        )
    def forward(self, x):
        return self.projector(x)


class Encoder_CLIP(nn.Module):
    def __init__(self, clip_dim=512, hdim=256, feat_dim=32):
        """
        Encoder that maps CLIP features to a hidden representation and
        produces four latent codes: z_x, z_y, z_s, z_r.
        """
        super(Encoder_CLIP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(clip_dim, hdim),
            nn.ReLU(True)
        )
        self.proj_x = Project(hdim, 2 * (hdim - 3 * feat_dim))
        self.proj_y = Project(hdim, 2 * feat_dim)
        self.proj_r = Project(hdim, 2 * feat_dim)
        self.proj_s = Project(hdim, 2 * feat_dim)
    def forward(self, x):
        y = self.fc(x)
        mu_x, logvar_x = self.proj_x(y).chunk(2, dim=1)
        mu_y, logvar_y = self.proj_y(y).chunk(2, dim=1)
        mu_r, logvar_r = self.proj_r(y).chunk(2, dim=1)
        mu_s, logvar_s = self.proj_s(y).chunk(2, dim=1)
        z_x = self.reparameterize(mu_x, logvar_x)
        z_y = self.reparameterize(mu_y, logvar_y)
        z_r = self.reparameterize(mu_r, logvar_r)
        z_s = self.reparameterize(mu_s, logvar_s)
        return (z_x, z_y, z_s, z_r), (mu_x, mu_y, mu_s, mu_r), (logvar_x, logvar_y, logvar_s, logvar_r)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Decoder_CLIP(nn.Module):
    def __init__(self, latent_dim, clip_dim=512, hidden_dim=256):
        """
        Decoder that reconstructs the CLIP feature vector from the latent vector.
        """
        super(Decoder_CLIP, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, clip_dim)
        )
    def forward(self, z):
        return self.decoder(z)


class VAE_CLIPIntegrated(nn.Module):
    def __init__(self, clip_model, clip_dim=512, hdim=512, feat_dim=32):
        """
        Integrated VAE:
          - Receives raw images as input.
          - Uses the pretrained (frozen) CLIP model to extract features.
          - Encoder encodes the CLIP features into multiple latent codes.
          - Decoder reconstructs the CLIP feature vector.
        """
        super(VAE_CLIPIntegrated, self).__init__()
        self.clip_model = clip_model  # pretrained CLIP model (frozen)
        self.encoder = Encoder_CLIP(clip_dim, hdim, feat_dim)
        latent_dim = hdim  # (hdim - 3*feat_dim) + 3*feat_dim = hdim
        self.decoder = Decoder_CLIP(latent_dim, clip_dim, hidden_dim=hdim)
    def forward(self, images):
        # Resize images to CLIP's expected resolution if necessary.
        expected_res = self.clip_model.visual.input_resolution  # usually 224
        if images.shape[-1] != expected_res:
            images = F.interpolate(images, size=(expected_res, expected_res), mode='bilinear', align_corners=False)
        # Extract CLIP features (freeze gradients).
        with torch.no_grad():
            orig_features = self.clip_model.encode_image(images).float()
        (z_x, z_y, z_s, z_r), mus, logvars = self.encoder(orig_features)
        # Concatenate latent codes along feature dimension.
        z = torch.cat((z_x, z_y, z_s, z_r), dim=1)
        recon = self.decoder(z)
        return recon, (z_x, z_y, z_s, z_r), mus, logvars, orig_features


#########################################
# Define a simple Classifier (used for both update and opponent branches)
#########################################
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)  # output: single logit
        )
    def forward(self, x):
        return self.fc(x)
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]

#########################################
# Setup: Load pretrained CLIP model and freeze its parameters.
#########################################
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)
for param in clip_model.parameters():
    param.requires_grad = False

#########################################
# Model Initialization
#########################################
hdim = 512
feat_dim = args.feat_dim  # from args

vae_model = nn.DataParallel(VAE_CLIPIntegrated(clip_model, clip_dim=512, hdim=hdim, feat_dim=feat_dim)).cuda()

# Define classifiers as in your original code.
cls_y_zyr = nn.DataParallel(Classifier(input_dim=2 * feat_dim)).cuda()
cls_s_zsr = nn.DataParallel(Classifier(input_dim=2 * feat_dim)).cuda()
cls_y_zsr = nn.DataParallel(Classifier(input_dim=feat_dim)).cuda()
cls_s_zyr = nn.DataParallel(Classifier(input_dim=feat_dim)).cuda()
Dis = nn.DataParallel(Classifier(input_dim=3 * feat_dim)).cuda()  # for [z_y, z_r, z_s]

start_epoch = 0
if args.resume:
    # Load checkpoint for VAE and classifiers.
    vae_checkpoint = torch.load(os.path.join(args.save_dir, 'vae.pth'))
    vae_model.load_state_dict(vae_checkpoint['state_dict'])
    cls_y_zyr.load_state_dict(torch.load(os.path.join(args.save_dir, 'cls_y_zyr.pth'))['state_dict'])
    cls_s_zsr.load_state_dict(torch.load(os.path.join(args.save_dir, 'cls_s_zsr.pth'))['state_dict'])
    cls_y_zsr.load_state_dict(torch.load(os.path.join(args.save_dir, 'cls_y_zsr.pth'))['state_dict'])
    cls_s_zyr.load_state_dict(torch.load(os.path.join(args.save_dir, 'cls_s_zyr.pth'))['state_dict'])
    Dis.load_state_dict(torch.load(os.path.join(args.save_dir, 'dis.pth'))['state_dict'])
    start_epoch = torch.load(os.path.join(args.save_dir, 'cls_y_zyr.pth'))['epoch'] + 1

vae_param_lst = [{'params': vae_model.module.encoder.parameters()},
                 {'params': vae_model.module.decoder.parameters(), 'lr': 5e-4}]
cls_update_param_lst = cls_y_zyr.module.get_parameters() + cls_s_zsr.module.get_parameters()
cls_opponent_param_lst = cls_y_zsr.module.get_parameters() + cls_s_zyr.module.get_parameters()

optimizer_vae = torch.optim.Adam(vae_param_lst, lr=1e-4, weight_decay=1e-5)
optimizer_cls_update = torch.optim.Adam(cls_update_param_lst, lr=1e-4, weight_decay=1e-5)
optimizer_cls_opponent = torch.optim.Adam(cls_opponent_param_lst, lr=1e-4, weight_decay=1e-5)
optimizer_d = torch.optim.Adam(Dis.module.get_parameters(), lr=1e-4, weight_decay=1e-5)

train_set, valid_set, test_set = ImageLoader(dataname=args.dataname)
transform, transform_test = load_transform(dataname=args.dataname)

train_data = ImageDataset(train_set, args.dataname, 'Male', args.label, './data/CelebA/img_align_celeba'
                              , transform)

valid_data = ImageDataset(valid_set, args.dataname, 'Male', args.label, './data/CelebA//img_align_celeba'
                          , transform_test)

test_data = ImageDataset(test_set, args.dataname, 'Male', args.label, './data/CelebA//img_align_celeba'
                         , transform_test)

bs = args.bs
bs_val = 10
bs_test = 100

trainloader = DataLoader(train_data, batch_size=bs, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
validloader = DataLoader(valid_data, batch_size=bs_val, shuffle=False, num_workers=8, pin_memory=True,
                         persistent_workers=True)
testloader = DataLoader(test_data, batch_size=bs_test, shuffle=False, num_workers=8, pin_memory=True,
                        persistent_workers=True)

# Prepare a validation batch (assume validloader is defined)
valid_iter = iter(validloader)
x_valid, s_valid, y_valid = next(valid_iter)
x_valid = x_valid.cuda()
s_valid = s_valid.cuda().float().view(-1, 1)
y_valid = y_valid.cuda().float().view(-1, 1)

#########################################
# Training Loop
#########################################
for epoch in range(start_epoch, args.epochs + 1):

    step = 0.
    loss_elbo_hist = 0.
    loss_recon_hist = 0.
    loss_prior_hist = 0.
    loss_fair_hist_I_y_y_given_zr = 0.
    loss_fair_hist_I_s_s_given_zr = 0.
    loss_y_hist = 0.
    loss_s_hist = 0.
    loss_yop_hist = 0.
    loss_sop_hist = 0.
    loss_tc_hist = 0.
    iters = tqdm(trainloader)

    vae_model.train()
    cls_y_zyr.train()
    cls_s_zsr.train()
    cls_y_zsr.train()
    cls_s_zyr.train()
    Dis.train()

    for x_batch, s_batch, y_batch in iters:
        step += 1
        x_batch = x_batch.cuda()
        s_batch = s_batch.cuda().float().view(-1, 1)
        y_batch = y_batch.cuda().float().view(-1, 1)

        # Clear gradients
        optimizer_vae.zero_grad()
        optimizer_cls_update.zero_grad()
        optimizer_cls_opponent.zero_grad()

        # Forward pass: integrated VAE internally extracts CLIP features.
        recon, (z_x, z_y, z_s, z_r), mus, logvars, orig_features = vae_model(x_batch)
        # Concatenate latent means and logvars for KL divergence
        mu = torch.cat(mus, dim=1)
        logvar = torch.cat(logvars, dim=1)
        z_all = torch.cat((z_x, z_y, z_s, z_r), dim=1)

        # Reconstruction loss (L1 loss on CLIP features)
        loss_recon = F.l1_loss(recon, orig_features, reduction='sum') / x_batch.size(0)
        # KL divergence loss
        loss_prior = -0.5 * (1 + logvar - mu.pow(2) - torch.clamp(logvar.exp(), min=1e-6, max=1000.0)).sum() / x_batch.size(0)
        loss_elbo = loss_recon + loss_prior

        # Classification loss: update branch using z_y and z_r for Y; z_s and z_r for S.
        pred_y_update = cls_y_zyr(torch.cat([z_y, z_r], dim=1))
        pred_s_update = cls_s_zsr(torch.cat([z_s, z_r], dim=1))
        loss_y_update = F.binary_cross_entropy_with_logits(pred_y_update, y_batch)
        loss_s_update = F.binary_cross_entropy_with_logits(pred_s_update, s_batch)
        cls_loss_update = (2 * loss_y_update + loss_s_update)

        # Opponent classification loss
        pred_y_opponent = cls_y_zsr(z_s)
        pred_s_opponent = cls_s_zyr(z_y)
        loss_y_opponent = F.binary_cross_entropy_with_logits(pred_y_opponent, y_batch)
        loss_s_opponent = F.binary_cross_entropy_with_logits(pred_s_opponent, s_batch)
        cls_loss_opponent = args.lambda_cls * (loss_y_opponent + loss_s_opponent)

        # Mutual information losses (assumed to be computed via provided functions)
        I_y_s_given_zr, I_s_y_given_zr = direct_CMI_loss(cls_y_zsr, cls_s_zyr, z_y, z_s)
        I_y_y_given_zr, I_s_s_given_zr = learning_relevant_zr(cls_y_zyr, cls_s_zsr,
                                                              z_y.detach(), z_r, z_s.detach(),
                                                              y_batch, s_batch,
                                                              alpha=args.alpha, beta=args.beta)

        # Discriminator loss
        pred_d = torch.sigmoid(Dis(torch.cat([z_y, z_r, z_s], dim=1)))
        loss_D = torch.log(torch.clamp(pred_d / torch.clamp(1 - pred_d, min=1e-7), min=1e-7)).mean()

        loss = loss_elbo + args.lambda_cls * cls_loss_update + args.lambda_tc * loss_D

        loss_fair_1 = args.lambda_fair_yzs * I_y_s_given_zr
        loss_fair_2 = args.lambda_fair_szy * I_s_y_given_zr
        loss_fair_3 = args.lambda_fair_yzy * -I_y_y_given_zr
        loss_fair_4 = args.lambda_fair_szs * -I_s_s_given_zr

        # Temporarily freeze update branch classifiers
        for params in cls_y_zyr.module.parameters():
            params.requires_grad = False
        for params in cls_s_zsr.module.parameters():
            params.requires_grad = False
        loss_fair_3.backward(retain_graph=True)
        loss_fair_4.backward(retain_graph=True)
        for params in cls_y_zyr.module.parameters():
            params.requires_grad = True
        for params in cls_s_zsr.module.parameters():
            params.requires_grad = True

        # Update encoder with CMI loss (if epoch reached threshold)
        if epoch >= args.start_CMI_epoch:
            for params in cls_y_zsr.module.parameters():
                params.requires_grad = False
            for params in cls_s_zyr.module.parameters():
                params.requires_grad = False
            loss_fair_1.backward(retain_graph=True)
            loss_fair_2.backward(retain_graph=True)
            for params in cls_y_zsr.module.parameters():
                params.requires_grad = True
            for params in cls_s_zyr.module.parameters():
                params.requires_grad = True

        # Opponent classification loss: freeze encoder update temporarily.
        for params in vae_model.module.encoder.parameters():
            params.requires_grad = False
        cls_loss_opponent.backward(retain_graph=True)
        for params in vae_model.module.encoder.parameters():
            params.requires_grad = True

        loss.backward(retain_graph=True)
        optimizer_vae.step()
        optimizer_cls_update.step()
        optimizer_cls_opponent.step()

        loss_recon_hist += loss_recon.item()
        loss_prior_hist += loss_prior.item()
        loss_fair_hist_I_y_y_given_zr += I_y_y_given_zr.item()
        loss_fair_hist_I_s_s_given_zr += I_s_s_given_zr.item()
        loss_y_hist += loss_y_update.item()
        loss_s_hist += loss_s_update.item()
        loss_yop_hist += loss_y_opponent.item()
        loss_sop_hist += loss_s_opponent.item()
        loss_tc_hist += loss_D.item()

        # --- Train Discriminator ---
        recon, (z_x, z_y, z_s, z_r), mus, logvars, orig_features = vae_model(x_batch)
        pred_d_real = torch.sigmoid(Dis(torch.cat([z_y, z_r, z_s], dim=1)))
        pred_d_neg = torch.sigmoid(Dis(permute_zs([z_y, z_r, z_s])))
        D_tc_loss = args.lambda_cls * (
            - torch.log(pred_d_real + 1e-7).mean()
            - torch.log(1 - pred_d_neg + 1e-7).mean()
        )
        optimizer_d.zero_grad()
        D_tc_loss.backward()
        optimizer_d.step()

        iters.set_description(
            'epoch: {},Recon: {:.3f},Prior: {:.3f},I_yy_zr: {:.3f},I_ss_zr: {:.3f},TC loss: {:.3f},Y loss: {:.3f},S loss: {:.3f},Yop loss: {:.3f},Sop loss: {:.3f}'
            .format(epoch, loss_recon_hist / step, loss_prior_hist / step, loss_fair_hist_I_y_y_given_zr / step,
                    loss_fair_hist_I_s_s_given_zr / step,
                    loss_tc_hist / step,
                    loss_y_hist / step, loss_s_hist / step, loss_yop_hist / step, loss_sop_hist / step)
        )

    write_loss_to_log(
        '\nepoch: {},Recon: {:.3f},Prior: {:.3f},I_yy_zr: {:.3f},I_ss_zr: {:.3f},TC loss: {:.3f},Y loss: {:.3f},S loss: {:.3f},Yop loss: {:.3f},Sop loss: {:.3f}'
        .format(epoch, loss_recon_hist / step, loss_prior_hist / step, loss_fair_hist_I_y_y_given_zr / step,
                loss_fair_hist_I_s_s_given_zr / step,
                loss_tc_hist / step,
                loss_y_hist / step, loss_s_hist / step, loss_yop_hist / step, loss_sop_hist / step)
    )

    # ------------------ Evaluation Code ------------------
    # Evaluate every 2 epochs when epoch >= 10 using test dataset.
    if epoch % 2 == 0 and epoch >= 10:
        # Set models to evaluation mode.
        vae_model.eval()
        cls_y_zyr.eval()
        cls_s_zsr.eval()
        cls_y_zsr.eval()
        cls_s_zyr.eval()

        with torch.no_grad():
            total_test = 0
            correct_y_test = 0
            correct_s_test = 0

            # Iterate over the test dataset.
            for x_test_batch, s_test_batch, y_test_batch in testloader:
                x_test_batch = x_test_batch.cuda()
                s_test_batch = s_test_batch.cuda().float().view(-1, 1)
                y_test_batch = y_test_batch.cuda().float().view(-1, 1)

                # Forward pass: integrated VAE extracts CLIP features and outputs latent codes.
                # Here, we only use z_y, z_s, z_r for classification.
                _, (z_x_test, z_y_test, z_s_test, z_r_test), _, _, _ = vae_model(x_test_batch)

                # Predict Y using update branch: combine z_y and z_r.
                pred_y_test = cls_y_zyr(torch.cat([z_y_test, z_r_test], dim=1))
                pred_y_test = torch.sigmoid(pred_y_test)
                pred_y_test = (pred_y_test >= 0.5).float()

                # Predict S using update branch: combine z_s and z_r.
                pred_s_test = cls_s_zsr(torch.cat([z_s_test, z_r_test], dim=1))
                pred_s_test = torch.sigmoid(pred_s_test)
                pred_s_test = (pred_s_test >= 0.5).float()

                total_test += y_test_batch.size(0)
                correct_y_test += (pred_y_test == y_test_batch).sum().item()
                correct_s_test += (pred_s_test == s_test_batch).sum().item()

            test_acc_y = correct_y_test / total_test
            test_acc_s = correct_s_test / total_test

            print(f"[Test set evaluation at epoch {epoch}] AccY: {test_acc_y:.3f}, AccA: {test_acc_s:.3f}")
            write_loss_to_log(f"[Test set evaluation at epoch {epoch}] AccY: {test_acc_y:.3f}, AccA: {test_acc_s:.3f}")

    # ------------------ Save Checkpoints ------------------
    # Save model checkpoints every 10 epochs.
    if epoch % 10 == 0:
        torch.save({'state_dict': vae_model.module.encoder.state_dict(), 'epoch': epoch},
                   os.path.join(save_dir, f'vae_encoder_{epoch}.pth'))
        torch.save({'state_dict': vae_model.module.decoder.state_dict(), 'epoch': epoch},
                   os.path.join(save_dir, f'vae_decoder_{epoch}.pth'))
        torch.save({'state_dict': cls_y_zyr.state_dict(), 'epoch': epoch},
                   os.path.join(save_dir, f'cls_y_zyr_{epoch}.pth'))
        torch.save({'state_dict': cls_s_zsr.state_dict(), 'epoch': epoch},
                   os.path.join(save_dir, f'cls_s_zsr_{epoch}.pth'))
        torch.save({'state_dict': cls_y_zsr.state_dict(), 'epoch': epoch},
                   os.path.join(save_dir, f'cls_y_zsr_{epoch}.pth'))
        torch.save({'state_dict': cls_s_zyr.state_dict(), 'epoch': epoch},
                   os.path.join(save_dir, f'cls_s_zyr_{epoch}.pth'))
        torch.save({'state_dict': Dis.state_dict(), 'epoch': epoch},
                   os.path.join(save_dir, f'dis_{epoch}.pth'))

