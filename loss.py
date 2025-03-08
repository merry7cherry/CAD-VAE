from typing import Callable

import torch
import torch.nn.functional as F


def permute_zs(zs):
    B, _ = zs[0].size()
    perm_z = []

    for z_i in zs:
        perm = torch.randperm(B).cuda()
        perm_z.append(z_i[perm])
    return torch.cat(perm_z, 1)


def learning_relevant_zr(cls_y_zyr: Callable, cls_s_zsr: Callable, z_y: torch.Tensor, z_r: torch.Tensor, z_s: torch.Tensor,
                      y_batch: torch.Tensor, s_batch: torch.Tensor,
                      alpha: float = 0.5, beta: float = 0.5) -> (torch.Tensor, torch.Tensor):
    """
    Compute the Learning Relevance loss (L_LRI).

    Args:
        cls_y_zyr (Callable): A classifier function Y = (ZY, ZR).
        cls_s_zsr (Callable): A classifier function S = (ZS, ZR).
        z_y (torch.Tensor): Feature tensor of shape (N, D).
        z_r (torch.Tensor): Feature tensor of shape (N, D).
        z_s (torch.Tensor): Feature tensor of shape (N, D).
        y_batch (torch.Tensor): Binary indicator tensor of shape (N,).
        s_batch (torch.Tensor): Binary indicator tensor of shape (N,).
        alpha (float, optional): Trade-off parameter. Default is 0.5.
        beta (float, optional): Trade-off parameter. Default is 0.5.

    Returns:
        torch.Tensor: Computed LRI loss.
    """
    N = z_y.shape[0]
    assert z_y.shape[0] == z_r.shape[0] == z_s.shape[0] == y_batch.shape[0] == s_batch.shape[0] \
        , "Input tensor size mismatch."

    # ========== I(Y;y|zr) ===========
    # -------- H(y | zr) using cls_y_zyr(zy, zr) ----------
    # Expand and concatenate z_y and z_r
    z_y_repeat = z_y.unsqueeze(1).expand(-1, N, -1)  # (N, N, D1)
    z_r_repeat = z_r.unsqueeze(0).expand(N, -1, -1)  # (N, N, D2)

    z_yr = torch.cat([z_y_repeat, z_r_repeat], dim=-1).view(N ** 2, -1)  # (N^2, D1 + D2)

    # Compute probability estimates
    p_y = torch.sigmoid(cls_y_zyr(z_yr)).view(N, N, -1)  # (N, N, 1)

    p_y_zr = p_y.mean(dim=0)  # (N, 1), average over different z_y
    # Compute conditional entropy H(y | zr)
    H_y_cond_zr = -(p_y_zr * torch.log(torch.clamp(p_y_zr, min=1e-7)) +
                    (1 - p_y_zr) * torch.log(torch.clamp(1 - p_y_zr, min=1e-7))).mean()

    # -------- H(y | zy, y) using cls_y_zyr(zy, zr)  ----------
    # Compute conditional entropy H(y | zr, s)
    y_idx = y_batch.bool().view(-1)  # Convert to boolean mask

    p_y_zr_y1 = p_y[y_idx, :][:, y_idx, :].mean(dim=0)
    p_y_zr_y0 = p_y[~y_idx, :][:, ~y_idx, :].mean(dim=0)

    H_y_cond_zr_y = -(
            (p_y_zr_y1 * torch.log(torch.clamp(p_y_zr_y1, min=1e-7)) +
             (1 - p_y_zr_y1) * torch.log(torch.clamp(1 - p_y_zr_y1, min=1e-7))).sum() +
            (p_y_zr_y0 * torch.log(torch.clamp(p_y_zr_y0, min=1e-7)) +
             (1 - p_y_zr_y0) * torch.log(torch.clamp(1 - p_y_zr_y0, min=1e-7))).sum()
    ) / N

    # Compute mutual information term
    I_y_y_given_zr = alpha * H_y_cond_zr - H_y_cond_zr_y

    # ========== I(S;s|zr) ===========
    # -------- H(s | zr) using cls_s_zsr(zs, zr) ----------
    # Expand and concatenate z_r and z_y
    z_s_repeat = z_s.unsqueeze(1).expand(-1, N, -1)  # (N, N, D1)
    z_r_repeat = z_r.unsqueeze(0).expand(N, -1, -1)  # (N, N, D2)

    z_sr = torch.cat([z_s_repeat, z_r_repeat], dim=-1).view(N ** 2, -1)  # (N^2, D1 + D2)

    # Compute probability estimates
    p_s = torch.sigmoid(cls_s_zsr(z_sr)).view(N, N, -1)  # (N, N, 1)
    p_s_zr = p_s.mean(dim=0)  # (N, 1), average over different z_s

    # Compute conditional entropy H(y | z)
    H_s_cond_zr = -(p_s_zr * torch.log(torch.clamp(p_s_zr, min=1e-7)) +
                    (1 - p_s_zr) * torch.log(torch.clamp(1 - p_s_zr, min=1e-7))).mean()

    # -------- H(s | zs, s) using cls_s_zsr(zs, zr) ----------
    # Compute conditional entropy H(s | zr, y)
    s_idx = s_batch.bool().view(-1)  # Convert to boolean mask

    p_s_zr_s1 = p_s[s_idx, :][:, s_idx, :].mean(dim=0)
    p_s_zr_s0 = p_s[~s_idx, :][:, ~s_idx, :].mean(dim=0)

    H_s_cond_zr_s = -(
            (p_s_zr_s1 * torch.log(torch.clamp(p_s_zr_s1, min=1e-7)) +
             (1 - p_s_zr_s1) * torch.log(torch.clamp(1 - p_s_zr_s1, min=1e-7))).sum() +
            (p_s_zr_s0 * torch.log(torch.clamp(p_s_zr_s0, min=1e-7)) +
             (1 - p_s_zr_s0) * torch.log(torch.clamp(1 - p_s_zr_s0, min=1e-7))).sum()
    ) / N

    # Compute mutual information term
    I_s_s_given_zr = beta * H_s_cond_zr - H_s_cond_zr_s

    return I_y_y_given_zr, I_s_s_given_zr


def direct_CMI_loss(cls_y_zsr, cls_s_zyr, z_y, z_s):
    r"""
        Computes:
          I_phi(\hat{Y}; \hat{S} | z_R), I_phi(\hat{S}; \hat{Y} | z_R)

        with:
          I_phi(\hat{Y}; \hat{S} | z_R) = H_phi(\hat{Y} | z_r) - H_phi(\hat{Y} | z_s, z_r),
          I_phi(\hat{S}; \hat{Y} | z_R) = H_phi(\hat{S} | z_r) - H_phi(\hat{S} | z_y, z_r),

        Arguments:
          cls_y_zsr:     a network that takes [z_s] => outputs single logit for y=1.
          cls_s_zyr:     a network that takes [z_y] => outputs single logit for s=1.
          z_y:       latent code for Y, shape [N, dim_z_y]
          z_s:       latent code for S, shape [N, dim_z_s]

        Returns:
          A scalar (float Tensor), representing
             H(Y|Z_S), H(S|Z_Y)
    """

    # ======== Compute H(Y|Z_S) ========
    # => pass to cls_y_zsr(opponent)
    p_y_zs = torch.sigmoid(cls_y_zsr(z_s)).view(z_y.size(0), -1)  # [N, 1]
    H_y_zs = -(
            p_y_zs * torch.log(torch.clamp(p_y_zs, min=1e-7))
            + (1 - p_y_zs) * torch.log(torch.clamp(1 - p_y_zs, min=1e-7))
    ).mean()

    # => I(Y; z_s|z_R) = H(Y|z_R) - H(Y|z_s, z_R)
    I_y_s_given_zr = - H_y_zs

    # ======== Compute H(S|Z_Y) ========
    # => pass to cls_s_zyr(opponent)
    p_s_zy = torch.sigmoid(cls_s_zyr(z_y)).view(z_s.size(0), -1)  # [N, 1]
    H_s_zy = -(
            p_s_zy * torch.log(torch.clamp(p_s_zy, min=1e-7))
            + (1 - p_s_zy) * torch.log(torch.clamp(1 - p_s_zy, min=1e-7))
    ).mean()

    # => I(S; z_y|z_R) = H(S|z_R) - H(S|z_y, z_R)
    I_s_y_given_zr = - H_s_zy

    # -----------------------------------------------------
    # I(Y; z_s|z_R), I(S; z_y|z_R)
    # -----------------------------------------------------
    return I_y_s_given_zr, I_s_y_given_zr
