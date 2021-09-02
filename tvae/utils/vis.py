import torchvision
import os
import wandb 
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def plot_recon(x, xhat, s_dir, e, wandb_on):
    x_path = os.path.join(s_dir, f'{e}_x.png')
    xhat_path = os.path.join(s_dir, f'{e}_xrecon.png')
    diff_path = os.path.join(s_dir, f'{e}_recon_diff.png')

    n_row = int(x.shape[0] ** 0.5)

    os.makedirs(s_dir, exist_ok=True)
    torchvision.utils.save_image(
        xhat, xhat_path, nrow=n_row,
        padding=2, normalize=False)

    torchvision.utils.save_image(
        x, x_path, nrow=n_row,
        padding=2, normalize=False)

    xdiff = torch.abs(x - xhat)

    torchvision.utils.save_image(
        xdiff, diff_path, nrow=n_row,
        padding=2, normalize=False)

    if wandb_on:
        wandb.log({'X Original':  wandb.Image(x_path)})
        wandb.log({'X Recon':  wandb.Image(xhat_path)})
        wandb.log({'Recon diff':  wandb.Image(diff_path)})


def plot_traversal_recon(x, s_dir, e, n_transforms, wandb_on, name='Cap_Traversal'):
    x_orig = x[0][0:n_transforms]
    x_recon = x[1][0:n_transforms]
    x_trav = [t[0].unsqueeze(0) for t in x[2:]]

    x_trav_path = os.path.join(s_dir, f'{e}_{name}.png')
    os.makedirs(s_dir, exist_ok=True)

    x_image = torch.cat([x_orig, x_recon] + x_trav)

    torchvision.utils.save_image(
        x_image, x_trav_path, nrow=n_transforms,
        padding=2, pad_value=1.0, normalize=False)

    if wandb_on:
        wandb.log({name:  wandb.Image(x_trav_path)})
    

def plot_filters(weight, name='TVAE_Filters', max_s=64, max_plots=3, wandb_on=True):
    weights_grid = weight.detach().cpu().numpy()
    empy_weight = np.zeros_like(weights_grid[0,0,:,:])
    c_out, c_in, h, w = weights_grid.shape
    s = min(max_s, int(np.ceil(np.sqrt(c_out))))

    if c_in == 3:
        f, axarr = plt.subplots(s,s)
        f.set_size_inches(min(s, 30), min(s, 30))
        empy_weight = np.zeros_like(weights_grid[0,:,:,:]).transpose((1, 2, 0))
        for s_h in range(s):
            for s_w in range(s):
                w_idx = s_h * s + s_w
                if w_idx < c_out:
                    filter_norm = colors.Normalize()(weights_grid[w_idx, :, :, :].transpose((1, 2, 0)))
                    img = axarr[s_h, s_w].imshow(filter_norm)
                    axarr[s_h, s_w].get_xaxis().set_visible(False)
                    axarr[s_h, s_w].get_yaxis().set_visible(False)
                    # f.colorbar(img, ax=axarr[s_h, s_w])
                else:
                    img = axarr[s_h, s_w].imshow(empy_weight)
                    axarr[s_h, s_w].get_xaxis().set_visible(False)
                    axarr[s_h, s_w].get_yaxis().set_visible(False)
                    # f.colorbar(img, ax=axarr[s_h, s_w])
        if wandb_on:
            wandb.log({"{}".format(name): wandb.Image(plt)}, commit=False)
        else:
            plt.savefig("{}".format(name))
        plt.close('all')
    else:
        for c in range(min(c_in, max_plots)):
            f, axarr = plt.subplots(s,s)
            f.set_size_inches(s, s)
            for s_h in range(s):
                for s_w in range(s):
                    w_idx = s_h * s + s_w
                    if w_idx < c_out:
                        img = axarr[s_h, s_w].imshow(weights_grid[w_idx, c, :, :], cmap='PuBu_r')
                        axarr[s_h, s_w].get_xaxis().set_visible(False)
                        axarr[s_h, s_w].get_yaxis().set_visible(False)
                        # f.colorbar(img, ax=axarr[s_h, s_w])
                    else:
                        img = axarr[s_h, s_w].imshow(empy_weight, cmap='PuBu_r')
                        axarr[s_h, s_w].get_xaxis().set_visible(False)
                        axarr[s_h, s_w].get_yaxis().set_visible(False)
                        # f.colorbar(img, ax=axarr[s_h, s_w])

            if wandb_on:
                wandb.log({"{}_cin{}".format(name, c): wandb.Image(plt)}, commit=False)
            else:
                plt.savefig("{}".format(name))
            plt.close('all')


def Plot_MaxActImg(all_s, all_x, s_dir, e, wandb_on):
    max_xs = []
    for s_idx in range(all_s.shape[1]):
        max_idx = torch.max(torch.abs(all_s[:, s_idx]), 0)[1]
        max_xs.append(all_x[max_idx].squeeze().unsqueeze(0).unsqueeze(0))
    
    path = os.path.join(s_dir, f'{e}_maxactimg.png')
    os.makedirs(s_dir, exist_ok=True)

    x_image = torch.cat(max_xs)

    sq = int(float(all_s.shape[1]) ** 0.5)

    torchvision.utils.save_image(
        x_image, path, nrow=sq,
        padding=2, normalize=False)

    if wandb_on:
        wandb.log({'Max_Act_Img':  wandb.Image(path)})

def Plot_ClassActMap(all_s, all_labels, s_dir, e, n_class=10, thresh=0.85, wandb_on=True):
    s_dim = all_s[0].shape[0]
    sq = int(float(s_dim)**0.5)
    class_selectivity = torch.ones_like(all_s[0]) * -1

    for c in range(n_class):
        s_c = all_s[all_labels.view(-1) == c]
        s_other = all_s[all_labels.view(-1) != c]

        s_mean_c = s_c.mean(0).squeeze()
        s_mean_other = s_other.mean(0).squeeze()
        s_var_c = s_c.var(0).squeeze()
        s_var_other = s_other.var(0).squeeze()

        dprime = (s_mean_c - s_mean_other) / torch.sqrt((s_var_c + s_var_other)/2.0)

        class_selectivity[dprime >= thresh] = c

    class_selectivity = class_selectivity.view(sq, sq)
    fig, ax = plt.subplots()
    ax.matshow(class_selectivity)
    for i in range(sq):
        for j in range(sq):
            c = class_selectivity[j,i].item()
            ax.text(i, j, str(int(c)), va='center', ha='center')

    if wandb_on:
        wandb.log({'Class Selectivity': wandb.Image(plt)})
    else:
        path = os.path.join(s_dir, f'{e}_classactmap.png')
        plt.savefig(path)
    plt.close('all')