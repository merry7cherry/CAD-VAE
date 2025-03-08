import os
import argparse
import shutil
import sys

# from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader import ImageLoader, ImageDataset, load_transform
from model import *
from loss import *

import torch.nn.functional as F

## final version

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=350)
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

args.log_name += 'CAD-VAE_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:.1f}_{:d}'.format(
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

dataname = f'{args.dataname}'

if args.dataname == 'celeba':
    dataname += f'_{args.label}'

save_dir = os.path.join('./save_CADVAE', dataname, args.log_name)

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


img_size = 128

train_set, valid_set, test_set = ImageLoader(dataname=args.dataname)
transform, transform_test = load_transform(dataname=args.dataname)

bs = args.bs
bs_val = 10
bs_test = 100

print(train_set)
print(valid_set)
print(test_set)

# Smiling or Attractive
if args.dataname == 'celeba':
    train_data = ImageDataset(train_set, args.dataname, 'Male', args.label, './data/CelebA/img_align_celeba'
                              , transform)

    valid_data = ImageDataset(valid_set, args.dataname, 'Male', args.label, './data/CelebA//img_align_celeba'
                              , transform_test)

    test_data = ImageDataset(test_set, args.dataname, 'Male', args.label, './data/CelebA//img_align_celeba'
                             , transform_test)

elif args.dataname == 'UTK':
    train_data = ImageDataset(train_set, 'UTK', 1, 0, transform=transform)
    valid_data = ImageDataset(valid_set, 'UTK', 1, 0, transform=transform)
    test_data = ImageDataset(test_set, 'UTK', 1, 0, transform=transform)

elif args.dataname == 'dnc':
    train_data = ImageDataset(train_set, 'dnc', transform=transform)
    valid_data = ImageDataset(valid_set, 'dnc', transform=transform)
    test_data = ImageDataset(test_set, 'dnc', transform=transform)

print(train_data)
print(valid_data)
print(test_data)

trainloader = DataLoader(train_data, batch_size=bs, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
validloader = DataLoader(valid_data, batch_size=bs_val, shuffle=False, num_workers=8, pin_memory=True,
                         persistent_workers=True)
testloader = DataLoader(test_data, batch_size=bs_test, shuffle=False, num_workers=8, pin_memory=True,
                        persistent_workers=True)

for images, sens, labels in trainloader:
    print(f"image size: {images.shape}")
    print(f"sensitive attribute: {sens}")
    print(f"label: {labels}")
    break

epochs = args.epochs
samples = 10

hdim = 512
feat_dim = args.feat_dim
channel = [64, 128, 256, 512, 512]
encoder = nn.DataParallel(Encoder_CADVAE(hdim=hdim, feat_dim=feat_dim, channels=channel, image_size=img_size)).cuda()
decoder = nn.DataParallel(Decoder_Res(hdim=hdim, channels=channel, image_size=img_size)).cuda()

# update classifier
cls_y_zyr = nn.DataParallel(Classifier(input_dim=2 * feat_dim)).cuda()
cls_s_zsr = nn.DataParallel(Classifier(input_dim=2 * feat_dim)).cuda()
# opponent classifier
cls_y_zsr = nn.DataParallel(Classifier(input_dim=1 * feat_dim)).cuda()
cls_s_zyr = nn.DataParallel(Classifier(input_dim=1 * feat_dim)).cuda()
# discriminator
Dis = nn.DataParallel(Classifier(input_dim=3 * feat_dim)).cuda()  # [z_y, z_r, z_s]

start_epoch = 0

if args.resume:
    encoder.load_state_dict(torch.load(os.path.join(save_dir, 'encoder.pth'))['state_dict'])
    decoder.load_state_dict(torch.load(os.path.join(save_dir, 'decoder.pth'))['state_dict'])
    cls_y_zyr.load_state_dict(torch.load(os.path.join(save_dir, 'cls_y_zyr.pth'))['state_dict'])
    cls_s_zsr.load_state_dict(torch.load(os.path.join(save_dir, 'cls_s_zsr.pth'))['state_dict'])
    cls_y_zsr.load_state_dict(torch.load(os.path.join(save_dir, 'cls_y_zsr.pth'))['state_dict'])
    cls_s_zyr.load_state_dict(torch.load(os.path.join(save_dir, 'cls_s_zyr.pth'))['state_dict'])
    Dis.load_state_dict(torch.load(os.path.join(save_dir, 'dis.pth'))['state_dict'])
    start_epoch = torch.load(os.path.join(save_dir, 'cls_y_zyr.pth'))['epoch'] + 1

vae_param_lst = [{'params': encoder.module.parameters()}, {'params': decoder.module.parameters(), 'lr': 5e-4}]
cls_update_param_lst = cls_y_zyr.module.get_parameters() + cls_s_zsr.module.get_parameters()
cls_opponent_param_lst = cls_y_zsr.module.get_parameters() + cls_s_zyr.module.get_parameters()

optimizer_vae = torch.optim.Adam(vae_param_lst, lr=1e-4, weight_decay=1e-5)
optimizer_cls_update = torch.optim.Adam(cls_update_param_lst, lr=1e-4, weight_decay=1e-5)
optimizer_cls_opponent = torch.optim.Adam(cls_opponent_param_lst, lr=1e-4, weight_decay=1e-5)
optimizer_d = torch.optim.Adam(Dis.module.get_parameters(), lr=1e-4, weight_decay=1e-5)

criterion = torch.nn.BCEWithLogitsLoss()
criterion_ce = torch.nn.CrossEntropyLoss()

valid_iter = iter(validloader)
x_valid, s_valid, y_valid = next(valid_iter)
x_valid, s_valid, y_valid = x_valid.cuda(), s_valid.cuda().float().view(-1, 1), y_valid.cuda().float().view(-1, 1)

for epoch in range(start_epoch, epochs + 1):

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

    encoder.train()
    decoder.train()
    cls_y_zyr.train()
    cls_s_zsr.train()
    cls_y_zsr.train()
    cls_s_zyr.train()
    Dis.train()

    for x_batch, s_batch, y_batch in iters:
        step += 1
        x_batch, s_batch, y_batch = x_batch.cuda(), s_batch.cuda().float().view(-1, 1), y_batch.cuda().float().view(-1,
                                                                                                                    1)
        # clear the gradient
        optimizer_vae.zero_grad()
        optimizer_cls_update.zero_grad()
        optimizer_cls_opponent.zero_grad()

        z, mu, logvar = encoder(x_batch)
        z_x, z_y, z_s, z_r = z
        mu_x, mu_y, mu_s, mu_r = mu
        logvar_x, logvar_y, logvar_s, logvar_r = logvar

        z = torch.cat(z, dim=-1)
        mu = torch.cat(mu, dim=-1)
        logvar = torch.cat(logvar, dim=-1)

        recon = decoder(z)

        ## ELBO loss
        loss_recon = F.l1_loss(recon, x_batch, reduction='sum') / z.shape[0]
        loss_prior = -0.5 * (1 + logvar - mu ** 2 - torch.clamp(logvar.exp(), min=1e-6, max=1000.0)).sum() / z.shape[0]

        loss_elbo = loss_prior + loss_recon

        # classification loss : update
        pred_y_update = cls_y_zyr(torch.cat([z_y, z_r], dim=-1))
        pred_s_update = cls_s_zsr(torch.cat([z_s, z_r], dim=-1))

        loss_y_update = criterion(pred_y_update, y_batch)
        loss_s_update = criterion(pred_s_update, s_batch)

        # classification loss to update encoder and classifier
        cls_loss_update = (2 * loss_y_update + loss_s_update)

        # classification loss : opponent
        pred_y_opponent = cls_y_zsr(torch.cat([z_s], dim=-1))
        pred_s_opponent = cls_s_zyr(torch.cat([z_y], dim=-1))

        loss_y_opponent = criterion(pred_y_opponent, y_batch)
        loss_s_opponent = criterion(pred_s_opponent, s_batch)

        # classification loss to update opponent classifier, freeze encoder
        cls_loss_opponent = args.lambda_cls * (loss_y_opponent + loss_s_opponent)

        # mutual information loss
        I_y_s_given_zr, I_s_y_given_zr = direct_CMI_loss(cls_y_zsr, cls_s_zyr, z_y, z_s)
        # learning relevance
        I_y_y_given_zr, I_s_s_given_zr = learning_relevant_zr(cls_y_zyr, cls_s_zsr, z_y.detach(), z_r, z_s.detach(), y_batch, s_batch,
                                                              alpha=args.alpha, beta=args.beta)

        ## Discriminator
        pred_d = torch.sigmoid(Dis(torch.cat([z_y, z_r, z_s], dim=-1)))  # [z_y, z_r, z_s]
        loss_D = torch.log(torch.clamp(pred_d / torch.clamp(1 - pred_d, min=1e-7), min=1e-7)).mean()

        loss = loss_elbo + args.lambda_cls * cls_loss_update + args.lambda_tc * loss_D

        loss_fair_1 = args.lambda_fair_yzs * I_y_s_given_zr
        loss_fair_2 = args.lambda_fair_szy * I_s_y_given_zr
        loss_fair_3 = args.lambda_fair_yzy * -I_y_y_given_zr
        loss_fair_4 = args.lambda_fair_szs * -I_s_s_given_zr

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

        # CMI loss update the encoder, freeze the cls
        if epoch >= args.start_CMI_epoch:  # epoch > start_CMI_epoch, introduce CMI loss
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

        # opponent classification do not optimize encoder, only update opponent classifier(cls_y_zsr, cls_s_zyr)
        # joint loss do not optimize encoder, the encoder optimization regarding ZX is due to exclude the Y and S information
        for params in encoder.module.parameters():
            params.requires_grad = False

        cls_loss_opponent.backward(retain_graph=True)

        for params in encoder.module.parameters():
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

        ## Train Discriminator
        z, mu, logvar = encoder(x_batch)
        z_x, z_y, z_s, z_r = z

        # Real samples: (z_y, z_r, z_s) from the true distribution
        pred_d_real = torch.sigmoid(Dis(torch.cat([z_y, z_r, z_s], dim=-1)))

        # Negative samples: independently permuting z_y and z_s
        pred_d_neg = torch.sigmoid(Dis(permute_zs([z_y, z_r, z_s])))

        # Total Correlation loss: force the discriminator to separate real and fake samples
        D_tc_loss = args.lambda_cls * (
                - torch.log(pred_d_real + 1e-7).mean()
                - torch.log(1 - pred_d_neg + 1e-7).mean()
        )

        optimizer_d.zero_grad()
        D_tc_loss.backward()
        optimizer_d.step()

        iters.set_description(
            'epoch: {},Recon: {:.3f},Prior: {:.3f},I_yy_zr: {:.3f},I_ss_zr: {:.3f},TC loss: {:.3f},Y loss: {:.3f},S loss: {:.3f},Yop loss: {:.3f},Sop loss: {:.3f}' \
                .format(epoch, loss_recon_hist / step, loss_prior_hist / step, loss_fair_hist_I_y_y_given_zr / step,
                        loss_fair_hist_I_s_s_given_zr / step,
                        loss_tc_hist / step,
                        loss_y_hist / step, loss_s_hist / step, loss_yop_hist / step, loss_sop_hist / step))

    write_loss_to_log(
        '\nepoch: {},Recon: {:.3f},Prior: {:.3f},I_yy_zr: {:.3f},I_ss_zr: {:.3f},TC loss: {:.3f},Y loss: {:.3f},S loss: {:.3f},Yop loss: {:.3f},Sop loss: {:.3f}' \
            .format(epoch, loss_recon_hist / step, loss_prior_hist / step, loss_fair_hist_I_y_y_given_zr / step,
                    loss_fair_hist_I_s_s_given_zr / step,
                    loss_tc_hist / step,
                    loss_y_hist / step, loss_s_hist / step, loss_yop_hist / step, loss_sop_hist / step))

    # evaluate every 10 epoch
    if epoch % 2 == 0 and epoch >= 10:
        encoder.eval()
        decoder.eval()
        cls_y_zyr.eval()
        cls_s_zsr.eval()
        cls_y_zsr.eval()
        cls_s_zyr.eval()

        with torch.no_grad():
            # result in train dataset
            pred_s_update = torch.sigmoid(pred_s_update)
            pred_s_update[pred_s_update >= 0.5] = 1
            pred_s_update[pred_s_update < 0.5] = 0
            pred_y_update = torch.sigmoid(pred_y_update)
            pred_y_update[pred_y_update >= 0.5] = 1
            pred_y_update[pred_y_update < 0.5] = 0

            acc_a = (pred_s_update == s_batch).float().mean()
            acc_y = (pred_y_update == y_batch).float().mean()

            print('epoch : {}, ELBO : {:.3f}, Y loss : {:.3f}, A loss : {:.3f}' \
                  .format(epoch, loss_elbo.item(), loss_y_update.item(), loss_s_update.item()))
            print('epoch : {}, Acc Y : {:.3f}, Acc A : {:.3f}'.format(epoch, acc_y, acc_a))

            # result in valid dataset
            total_test = 0
            correct_y_test = 0
            correct_s_test = 0

            for x_test_batch, s_test_batch, y_test_batch in testloader:
                x_test_batch = x_test_batch.cuda()
                s_test_batch = s_test_batch.cuda().float().view(-1, 1)
                y_test_batch = y_test_batch.cuda().float().view(-1, 1)

                # encode
                z_test, _, _ = encoder(x_test_batch)  # (z_x, z_y, z_s, z_r)
                z_x_test, z_y_test, z_s_test, z_r_test = z_test

                # Predict Y
                pred_y_test = cls_y_zyr(torch.cat([z_y_test, z_r_test], dim=-1))  # logits
                pred_y_test = torch.sigmoid(pred_y_test)
                pred_y_test = (pred_y_test >= 0.5).float()  # 0/1

                # Predict S
                pred_s_test = cls_s_zsr(torch.cat([z_s_test, z_r_test], dim=-1))
                pred_s_test = torch.sigmoid(pred_s_test)
                pred_s_test = (pred_s_test >= 0.5).float()

                # count accuracy
                total_test += y_test_batch.size(0)
                correct_y_test += (pred_y_test == y_test_batch).sum().item()
                correct_s_test += (pred_s_test == s_test_batch).sum().item()

            test_acc_y = correct_y_test / total_test
            test_acc_s = correct_s_test / total_test

            print(f"[Test set evaluation at epoch {epoch}] "
                  f"AccY: {test_acc_y:.3f}, "
                  f"AccA: {test_acc_s:.3f}")

            write_loss_to_log(f"[Test set evaluation at epoch {epoch}] "
                              f"AccY: {test_acc_y:.3f}, "
                              f"AccA: {test_acc_s:.3f}")

            # draw grid picture that permute each latent code
            for sample_idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                label_lst, img_lst = [], []

                z, _, _ = encoder(x_valid)
                z_x, z_y, z_s, z_r = z
                z = torch.cat(z, dim=-1)

                img_lst.append(torchvision.utils.make_grid(x_valid[: samples], nrow=samples).cpu().permute(1, 2, 0))
                label_lst.append('Original')

                z_ent = torch.cat([z_y, z_s, z_r], dim=1)
                z = torch.cat([z_x, z_ent], dim=1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1, 2, 0))
                label_lst.append('$Recon$')

                z_ent = torch.cat([z_y, z_s, z_r], dim=1)
                z = torch.cat([z_x[sample_idx].unsqueeze(0).repeat(bs_val, 1), z_ent], dim=1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1, 2, 0))
                label_lst.append('$[z_x^{' + "({})".format(sample_idx) + '}, z_y, z_s, z_r]$')

                z_ent = torch.cat([z_y[sample_idx].unsqueeze(0).repeat(bs_val, 1), z_s, z_r], dim=1)
                z = torch.cat([z_x, z_ent], dim=1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1, 2, 0))
                label_lst.append('$[z_x, z_y^{' + "({})".format(sample_idx) + '}, z_s, z_r]$')

                z_ent = torch.cat([z_y, z_s[sample_idx].unsqueeze(0).repeat(bs_val, 1), z_r], dim=1)
                z = torch.cat([z_x, z_ent], dim=1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1, 2, 0))
                label_lst.append('$[z_x, z_y, z_s^{' + "({})".format(sample_idx) + '}, z_r]$')

                z_ent = torch.cat([z_y, z_s, z_r[sample_idx].unsqueeze(0).repeat(bs_val, 1)], dim=1)
                z = torch.cat([z_x, z_ent], dim=1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1, 2, 0))
                label_lst.append('$[z_x, z_y, z_s, z_r^{' + "({})".format(sample_idx) + '}]$')

                z_ent = torch.cat([z_y, z_s[sample_idx].unsqueeze(0).repeat(bs_val, 1),
                                   z_r[sample_idx].unsqueeze(0).repeat(bs_val, 1)], dim=1)
                z = torch.cat([z_x, z_ent], dim=1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1, 2, 0))
                label_lst.append(
                    '$[z_x, z_y, z_s^{' + "({})".format(sample_idx) + '}, z_r^{' + "({})".format(sample_idx) + '}]$')

                z_ent = torch.cat([z_y, z_s, z_r], dim=1)
                z = torch.cat([z_x, z_ent[sample_idx].unsqueeze(0).repeat(bs_val, 1)], dim=1)
                recon = decoder(z)
                img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1, 2, 0))
                label_lst.append('$[z_x, z_y^{' + "({})".format(sample_idx) + '}, z_s^{' + "({})".format(
                    sample_idx) + '}, z_r^{' + "({})".format(sample_idx) + '}]$')

                # z_ent = torch.cat([z_y, z_s, z_r], dim=1)
                # z = torch.cat([torch.randn_like(z_x).cuda(), z_ent], dim=1)
                # recon = decoder(z)
                # img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1, 2, 0))
                # label_lst.append('[$N, z_y, z_s, z_r]$')
                #
                # z_ent = torch.cat([torch.randn_like(z_y).cuda(), z_s, z_r], dim=1)
                # z = torch.cat([torch.randn_like(z_x).cuda(), z_ent], dim=1)
                # recon = decoder(z)
                # img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1, 2, 0))
                # label_lst.append('[$z_x, N, z_s, z_r]$')
                #
                # z_ent = torch.cat([z_y, torch.randn_like(z_s).cuda(), z_r], dim=1)
                # z = torch.cat([torch.randn_like(z_x).cuda(), z_ent], dim=1)
                # recon = decoder(z)
                # img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1, 2, 0))
                # label_lst.append('[$z_x, z_y, N, z_r]$')
                #
                # z_ent = torch.cat([z_y, z_s, torch.randn_like(z_r).cuda()], dim=1)
                # z = torch.cat([z_x, z_ent], dim=1)
                # recon = decoder(z)
                # img_lst.append(torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1, 2, 0))
                # label_lst.append('[$z_x, z_y, z_s, N]$')

                # img_lst = [recon_1, recon_2, recon_3, recon_4, recon_5, recon_6, recon_7, recon_8, image_1]
                img = torch.cat(img_lst, 0).detach().numpy()

                img_h = x_valid.shape[2]
                img_w = int(x_valid.shape[3])

                label_list = []
                for i in range(samples):
                    tick = ''
                    if y_valid[i] == 1:
                        tick += f'{args.label.split("_")[0]}, '
                    else:
                        tick += f'Not {args.label.split("_")[0]}, '
                    if s_valid[i] == 1:
                        tick += 'Male'
                    else:
                        tick += 'Female'
                    label_list.append(tick)

                plt.figure(figsize=(24, 24))
                plt.imshow(img)
                plt.xticks([img_w * 0.5 + i * img_w for i in range(samples)], \
                           label_list)
                plt.yticks([img_h * 0.5 + i * img_h for i in range(len(img_lst))], label_lst, fontsize=17)
                plt.savefig(os.path.join(save_dir, 'recon_{}-{}.pdf'.format(epoch, sample_idx)), bbox_inches='tight')

        # save for well-trained model, save at every 10 epochs
        torch.save({'state_dict': encoder.state_dict(), 'epoch': epoch}, os.path.join(save_dir, f'encoder_{epoch}.pth'))
        torch.save({'state_dict': decoder.state_dict(), 'epoch': epoch}, os.path.join(save_dir, f'decoder_{epoch}.pth'))
        torch.save({'state_dict': cls_y_zyr.state_dict(), 'epoch': epoch},
                   os.path.join(save_dir, f'cls_y_zyr_{epoch}.pth'))
        torch.save({'state_dict': cls_s_zsr.state_dict(), 'epoch': epoch},
                   os.path.join(save_dir, f'cls_s_zsr_{epoch}.pth'))
        torch.save({'state_dict': cls_y_zsr.state_dict(), 'epoch': epoch},
                   os.path.join(save_dir, f'cls_y_zsr_{epoch}.pth'))
        torch.save({'state_dict': cls_s_zyr.state_dict(), 'epoch': epoch},
                   os.path.join(save_dir, f'cls_s_zyr_{epoch}.pth'))
        torch.save({'state_dict': Dis.state_dict(), 'epoch': epoch}, os.path.join(save_dir, f'dis_{epoch}.pth'))

    # save for resume, the new version will recover the old
    torch.save({'state_dict': encoder.state_dict(), 'epoch': epoch}, os.path.join(save_dir, 'encoder.pth'))
    torch.save({'state_dict': decoder.state_dict(), 'epoch': epoch}, os.path.join(save_dir, 'decoder.pth'))
    torch.save({'state_dict': cls_y_zyr.state_dict(), 'epoch': epoch}, os.path.join(save_dir, 'cls_y_zyr.pth'))
    torch.save({'state_dict': cls_s_zsr.state_dict(), 'epoch': epoch}, os.path.join(save_dir, 'cls_s_zsr.pth'))
    torch.save({'state_dict': cls_y_zsr.state_dict(), 'epoch': epoch}, os.path.join(save_dir, 'cls_y_zsr.pth'))
    torch.save({'state_dict': cls_s_zyr.state_dict(), 'epoch': epoch}, os.path.join(save_dir, 'cls_s_zyr.pth'))
    torch.save({'state_dict': Dis.state_dict(), 'epoch': epoch}, os.path.join(save_dir, 'dis.pth'))
