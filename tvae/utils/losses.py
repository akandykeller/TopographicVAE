import torch
import numpy as np
from numpy.fft import fft, ifft
from scipy import stats

def all_pairs_equivariance_loss_naive(s, bsz, seq_len, n_caps, cap_dim):
    s = s.view(bsz, seq_len, n_caps, cap_dim)
    s = s / s.norm(dim=-1, keepdim=True)
    eq_loss = 0.0
    for t0 in range(0, seq_len-1):
        for end_t in range(t0+1, seq_len):
            eq_loss += (s[:, t0].roll(shifts=end_t - t0, dims=-1) - s[:, end_t]).abs().sum() / bsz
    
    return eq_loss


def all_pairs_equivariance_loss(s, bsz, seq_len, n_caps, cap_dim):
    s = s.view(bsz, seq_len, n_caps, cap_dim)
    s = s / s.norm(dim=-1, keepdim=True)
    eq_loss = 0.0
    for t in range(1, seq_len):
        eq_loss += (s.roll(shifts=(t, t), dims=(1, -1)) - s).abs().sum() / bsz
    eq_loss = eq_loss / 2.0
    return eq_loss


def test_equivariance_loss(bsz=8, seq_len=18, n_caps=18, cap_dim=18):
    s = torch.randn((bsz, seq_len, n_caps, cap_dim))
    eq_naive = all_pairs_equivariance_loss_naive(s, bsz, seq_len, n_caps, cap_dim)
    eq_fast = all_pairs_equivariance_loss(s, bsz, seq_len, n_caps, cap_dim)
    assert torch.allclose(eq_naive, eq_fast)


def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.
    x and y must be real sequences with the same length.
    """
    return ifft(fft(x) * fft(y).conj()).real


def get_cap_offsets(cap_offsets, true_offsets, s, label, bsz, seq_len, n_caps, cap_dim):
    label = label.view(bsz, seq_len, -1)
    s = s.view(bsz, seq_len, n_caps, cap_dim).detach().cpu()
    transform_idxs = torch.argmax(torch.abs(label[:, 0] - label[:, 1]), dim=1)

    for b in range(bsz):
        transform_idx = transform_idxs[b].item()
        t0 = torch.where(label[b, :, transform_idx] == 0)[0]     

        cos = []
        tos = []
        diffs = []
        for i in t0:
            pcc = periodic_corr(s[b, 0], s[b, i])
            max_pcc_locs = np.argmax(pcc, axis=1)
            cos.append(stats.mode(max_pcc_locs)[0])
            tos.append(label[b, 0, transform_idx].item())
            diffs.append(abs(tos[-1]-cos[-1]))
        offset_idx = np.argmin(diffs)

        cap_offsets[transform_idx].append(cos[offset_idx][0])
        true_offsets[transform_idx].append(tos[offset_idx])

    return cap_offsets, true_offsets


if __name__ == '__main__':
    test_equivariance_loss()
    print("Equivariance Loss Test Passed")