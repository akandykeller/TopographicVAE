import torch
import matplotlib.pyplot as plt
import wandb 

def Plot_Covariance_Matrix(x, y, name='Covarince', fill_diagonal=True, wandb_on=True):
    X = x.view(x.shape[0], -1, 1).cpu().detach()
    Y = y.view(y.shape[0], -1, 1).cpu().detach()
    Sx = torch.std(X, dim=0)
    Sy = torch.std(Y, dim=0)
    Xm = X - X.mean(dim=0)
    Ym = Y - Y.mean(dim=0)
    C = Xm @ Ym.view(Ym.shape[0], 1, -1)
    C = C.mean(dim=0)
    S = Sx @ Sy.view(1, -1)
    C = C / S
    if fill_diagonal:
        C = C.fill_diagonal_(0.0)
    plt.imshow(C.numpy())
    if wandb_on:
        wandb.log({f"{name}": wandb.Image(plt)}, commit=False)
    else:
        plt.savefig(f"{name}")