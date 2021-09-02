import os
import torch
from tvae.utils.vis import plot_recon, Plot_MaxActImg, Plot_ClassActMap
from tvae.utils.correlations import Plot_Covariance_Matrix
from tvae.utils.losses import all_pairs_equivariance_loss, get_cap_offsets
import numpy as np

def train_epoch(model, optimizer, train_loader, log, savepath, epoch, eval_batches=300,
                plot_weights=False, plot_fullcaptrav=False, plot_samples=False, wandb_on=True):
    total_loss = 0
    total_kl = 0
    total_neg_logpx_z = 0
    total_eq_loss = 0.0
    num_batches = 0

    model.train()
    for x, label in train_loader:
        optimizer.zero_grad()
        x = x.float().to('cuda')

        x_batched = x.view(-1, *x.shape[-3:])  # batch transforms
        z, u, s, probs_x, kl_z, kl_u, neg_logpx_z = model(x_batched)

        avg_KLD = (kl_z.sum() + kl_u.sum()) / x_batched.shape[0]
        avg_neg_logpx_z = neg_logpx_z.sum() / x_batched.shape[0]
        loss = avg_neg_logpx_z + avg_KLD
        
        eq_loss = all_pairs_equivariance_loss(s.detach(), bsz=x.shape[0], 
                                              seq_len=x.shape[1], n_caps=model.grouper.n_caps,
                                              cap_dim=model.grouper.cap_dim)

        loss.backward()
        optimizer.step()    

        total_loss += loss
        total_neg_logpx_z += avg_neg_logpx_z
        total_kl += avg_KLD
        total_eq_loss += eq_loss
        num_batches += 1
        b_idx = epoch * len(train_loader) + num_batches

        if b_idx % eval_batches == 0:
            log('Train Total Loss', loss)
            log('Train -LogP(x|z)', avg_neg_logpx_z)
            log('Train KLD', avg_KLD)
            log('Eq Loss', eq_loss)

            if plot_weights:
                model.plot_decoder_weights(wandb_on=wandb_on)
                model.plot_encoder_weights(wandb_on=wandb_on)

            Plot_Covariance_Matrix(s**2.0, s**2.0, name='Covariance_S**2_batch', wandb_on=wandb_on)

            if plot_fullcaptrav:
                model.plot_capsule_traversal(x_batched.detach(), 
                                             os.path.join(savepath, 'samples'),
                                             b_idx, wandb_on=wandb_on,
                                             name='Cap_Traversal_Train')
            if plot_samples:
                model.plot_samples(x_batched, b_idx, os.path.join(savepath, 'samples'),
                                   n_samples=100, wandb_on=wandb_on)

            plot_recon(x_batched, 
                       probs_x.view(x_batched.shape), 
                       os.path.join(savepath, 'samples'),
                       b_idx, wandb_on=wandb_on)

    return total_loss, total_neg_logpx_z, total_kl, total_eq_loss, num_batches


def eval_epoch(model, val_loader, log, savepath, epoch, n_is_samples=100, 
               plot_maxact=False, plot_class_selectivity=False, 
               plot_cov=False, plot_fullcaptrav=True, wandb_on=True,
               cap_trav_batches=100):
    total_loss = 0
    total_kl = 0
    total_neg_logpx_z = 0
    total_is_estimate = 0.0
    total_eq_loss = 0.0
    num_batches = 0
    all_x = []
    all_s = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for x, label in val_loader:
            x = x.float().to('cuda')

            x_batched = x.view(-1, *x.shape[-3:])  # batch transforms
            z, u, s, probs_x, kl_z, kl_u, neg_logpx_z = model(x_batched)
            avg_KLD = (kl_z.sum() + kl_u.sum()) / x_batched.shape[0]
            avg_neg_logpx_z = neg_logpx_z.sum() / x_batched.shape[0]

            eq_loss = all_pairs_equivariance_loss(s, bsz=x.shape[0], seq_len=x.shape[1], 
                                                  n_caps=model.grouper.n_caps, cap_dim=model.grouper.cap_dim)

            all_s.append(s.cpu().detach())
            all_x.append(x.cpu().detach())
            all_labels.append(label.cpu().detach())

            loss = avg_neg_logpx_z + avg_KLD
            total_loss += loss
            total_neg_logpx_z += avg_neg_logpx_z
            total_kl += avg_KLD
            total_eq_loss += eq_loss

            is_estimate = model.get_IS_estimate(x_batched, n_samples=n_is_samples)
            total_is_estimate += is_estimate.sum() / x_batched.shape[0]

            num_batches += 1

            if plot_fullcaptrav and num_batches % cap_trav_batches == 0:
                model.plot_capsule_traversal(x_batched.detach(), 
                                os.path.join(savepath, 'samples'),
                                num_batches, wandb_on=wandb_on,
                                name='Cap_Traversal_Val')

    if plot_cov or plot_maxact or plot_class_selectivity:
        all_s = torch.cat(all_s, 0)
        all_x = torch.cat(all_x, 0)
        all_labels = torch.cat(all_labels, 0)
    if plot_cov:
        Plot_Covariance_Matrix(all_s, all_s, name='Covariance_S_Full', wandb_on=wandb_on)
        Plot_Covariance_Matrix(all_s**2.0, all_s**2.0, name='Covariance_S**2_Full', wandb_on=wandb_on)
    if plot_maxact:
        Plot_MaxActImg(all_s, all_x, os.path.join(savepath, 'samples'), epoch, wandb_on=wandb_on)
    if plot_class_selectivity:
        Plot_ClassActMap(all_s, all_labels, os.path.join(savepath, 'samples'), epoch, wandb_on=wandb_on)

    return total_loss, total_neg_logpx_z, total_kl, total_is_estimate, total_eq_loss, num_batches




def train_epoch_dsprites(model, optimizer, train_loader, log, savepath, epoch, eval_batches=300,
                         plot_weights=False, plot_fullcaptrav=False, 
                         compute_capcorr=False, wandb_on=True):
    total_loss = 0
    total_kl = 0
    total_neg_logpx_z = 0
    total_eq_loss = 0.0
    cap_offsets = {idx:[] for idx in range(1, 5)}
    true_offsets = {idx:[] for idx in range(1, 5)}
    num_batches = 0

    model.train()
    for x, label in train_loader:
        optimizer.zero_grad()
        x = x.float().to('cuda')

        x_batched = x.view(-1, *x.shape[-3:])  # batch transforms
        z, u, s, probs_x, kl_z, kl_u, neg_logpx_z = model(x_batched)

        avg_KLD = (kl_z.sum() + kl_u.sum()) / x_batched.shape[0]
        avg_neg_logpx_z = neg_logpx_z.sum() / x_batched.shape[0]
        loss = avg_neg_logpx_z + avg_KLD

        eq_loss = all_pairs_equivariance_loss(s.detach(), bsz=x.shape[0], seq_len=x.shape[1],
                                              n_caps=model.grouper.n_caps, cap_dim=model.grouper.cap_dim)
   
        if compute_capcorr:
            cap_offsets, true_offsets = get_cap_offsets(cap_offsets, true_offsets, s, 
                                                        label, bsz=x.shape[0], seq_len=x.shape[1], 
                                                        n_caps=model.grouper.n_caps, cap_dim=model.grouper.cap_dim)

        loss.backward()
        optimizer.step()    

        total_loss += loss
        total_neg_logpx_z += avg_neg_logpx_z
        total_kl += avg_KLD
        total_eq_loss += eq_loss
        num_batches += 1
        b_idx = epoch * len(train_loader) + num_batches

        if b_idx % eval_batches == 0:
            log('Train Total Loss', loss)
            log('Train -LogP(x|z)', avg_neg_logpx_z)
            log('Train KLD', avg_KLD)
            log('Eq Loss', eq_loss)

            if plot_weights:
                model.plot_decoder_weights(wandb_on=wandb_on)
                model.plot_encoder_weights(wandb_on=wandb_on)

            Plot_Covariance_Matrix(s**2.0, s**2.0, name='Covariance_S**2_batch', wandb_on=wandb_on)

            if plot_fullcaptrav:
                model.plot_capsule_traversal(x_batched.detach(), 
                                             os.path.join(savepath, 'samples'),
                                             b_idx, wandb_on=wandb_on)

            plot_recon(x_batched, 
                       probs_x.view(x_batched.shape), 
                       os.path.join(savepath, 'samples'),
                       b_idx, wandb_on=wandb_on)

            if compute_capcorr:
                for t in cap_offsets:
                    log(f'Train CapCorr {t}', np.corrcoef(cap_offsets[t], true_offsets[t])[0,1])

    total_cap_corr = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    if compute_capcorr:
        for t in cap_offsets:
            total_cap_corr[t] = np.corrcoef(cap_offsets[t], true_offsets[t])[0,1]

    return total_loss, total_neg_logpx_z, total_kl, total_eq_loss, total_cap_corr, num_batches