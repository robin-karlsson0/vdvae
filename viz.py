import imageio
import matplotlib.pyplot as plt
import torch
import numpy as np
from train_helpers import load_vaes, set_up_hyperparams
from data import set_up_data
from utils import (arrange_side_by_side, embmap2rgb_pca, image_grid, 
                   viz_rows2img, load_register, sample_inference)

def write_images(H, ema_vae, x_viz, x_oracle_viz, fname, logprint, temp=0.1):
    '''
    Args:
        x_viz: Float tensor (B,H,W,D)
        x_oracle_viz: Float tensor (B,H,W,D)
    '''
    # List for visualization rows
    batches = []

    ###########
    #  Input
    ###########
    zs = [
        s['z'].cuda() for s in ema_vae.forward_get_latents(x_viz,
                                                           mode='post_match')
    ]
    zs_oracle = [
        s['z'].cuda()
        for s in ema_vae.forward_get_latents(x_oracle_viz,
                                             mode='oracle')
    ]

    ###################
    #  Input RGB viz
    ###################
    # NOTE: Create a PCA object and reuse for all later RGB visualizations
    x_oracle_viz_rgb, pca = embmap2rgb_pca(x_oracle_viz)
    x_viz_rgb, _ = embmap2rgb_pca(x_viz, pca)
    input_viz = arrange_side_by_side(x_viz_rgb, x_oracle_viz_rgb)  # (B, H, W, 3)
    batches.append(input_viz)

    ################################
    #  Posterior sampling RGB viz
    ################################
    acts = ema_vae.encoder_post_match.forward(x_viz)
    px_z_post, _ = ema_vae.decoder.forward(acts, mode='post_match')
    px_z_post = torch.permute(px_z_post, (0, 2, 3, 1))

    acts = ema_vae.encoder.forward(x_oracle_viz)
    px_z_oracle, _ = ema_vae.decoder.forward(acts, mode='oracle')  # (B, D, H, W)
    px_z_oracle = torch.permute(px_z_oracle, (0, 2, 3, 1))

    px_post_viz_rgb, _ = embmap2rgb_pca(px_z_post, pca)
    px_oracle_viz_rgb, _ = embmap2rgb_pca(px_z_oracle, pca)
    px_viz = arrange_side_by_side(px_post_viz_rgb, px_oracle_viz_rgb)
    batches.append(px_viz)

    #############################
    #  Latent sampling RGB viz
    #############################
    # Visualize generations starting from different posterior distribution
    # latent layers (e.g. ]9, 19, ... , 56])
    viz_per_row = input_viz.shape[0]
    lv_points = np.floor(
        np.linspace(0, 1, H.num_variables_visualize + 2) *
        len(zs)).astype(int)[1:-1]
    
    for i in lv_points:
        latent_obs = ema_vae.forward_samples_set_latents(
            viz_per_row // 2,
            zs[:i],
            t=temp,
        )
        latent_oracle = ema_vae.forward_samples_set_latents(
            viz_per_row // 2,
            zs_oracle[:i],
            t=temp,
        )

        latent_obs = torch.permute(latent_obs, (0, 2, 3, 1))
        latent_obs_rgb, _ = embmap2rgb_pca(latent_obs, pca)
        latent_oracle = torch.permute(latent_oracle, (0, 2, 3, 1))
        latent_oracle_rgb, _ = embmap2rgb_pca(latent_oracle, pca)

        latent_viz = arrange_side_by_side(latent_obs_rgb, latent_oracle_rgb)
        batches.append(latent_viz)

    #######################
    #  Unconditional viz
    #######################
    viz_temps_list = H.viz_temps
    for t in viz_temps_list[:H.num_temperatures_visualize]:
        px_z = ema_vae.forward_uncond_samples(viz_per_row, t=t)
        px_z = torch.permute(px_z, (0, 2, 3, 1))
        px_z_rgb, _ = embmap2rgb_pca(px_z, pca)
        px_z_rgb = px_z_rgb.cpu().numpy()
        batches.append(px_z_rgb)

    logprint(f'printing samples to {fname}')

    img = viz_rows2img(batches)

    # Concatenate 'road' and 'intensity' visualizations to side-by-side img
    # im = np.concatenate([im[:, :, 0], im[:, :, 1]], axis=1)

    _, grid_h, grid_w, _ = x_viz.shape
    img = image_grid(img, grid_h, grid_w)

    imageio.imwrite(fname, img)


def viz_sample2rgb(sample_pth:str, vae, dataset, filename:str):
    '''
    Args:
        sample_pth: Path to a compressed sample .pkl.gz file.
    '''
    px_z = sample_inference(sample_pth, vae, dataset)
    px_z = torch.permute(px_z, (0, 2, 3, 1))
    px_z_rgb, _ = embmap2rgb_pca(px_z)

    plt.imshow(px_z_rgb.cpu().numpy()[0])
    plt.savefig(filename)


def viz_sample2sem(sample_pth:str, sem_txt:str, txt2idx_star:dict, idx_star2emb:dict, vae, dataset, filename:str, suff_sem_thresh:float=None):
    '''
    Args:
        sample_pth: Path to a compressed sample .pkl.gz file.
    '''
    px_z = sample_inference(sample_pth, vae, dataset)  # (H, W, D)

    sem_emb = idx_star2emb[txt2idx_star[sem_txt]]

    sim = torch.tensordot(px_z, sem_emb, dims=([0], [1]))
    sim = sim[:, :, 0]  # (H, W)

    if suff_sem_thresh is not None:
        suff_sim = sim > suff_sem_thresh
        plt.imshow(suff_sim.numpy())
    else:
        plt.imshow(sim.numpy())
        plt.colorbar()
    
    plt.savefig(filename)



if __name__ == '__main__':
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    vae, ema_vae = load_vaes(H, logprint)

    sample_pth = '/home/r_karlsson/workspace6/pc-accumulation-lib/pwm_carla_town10_3/subdir003/bev_605.pkl.gz'

    txt2idx_star = load_register('txt2idx_star_carla.pkl')
    idx_star2emb = load_register('idx_star2emb_carla_sbert.pkl')

    # viz_sample2rgb(sample_pth, ema_vae, data_valid_or_test, 'test.png')

    #sem_txt = 'road'
    sem_txt = 'road marking'
    suff_sem_thresh = None
    # suff_sem_thresh = 0.9*0.69284236  # road
    suff_sem_thresh = 0.9*0.68956494  # road marking
    viz_sample2sem(sample_pth, sem_txt, txt2idx_star, idx_star2emb, ema_vae, data_valid_or_test, 'test.png', suff_sem_thresh=suff_sem_thresh)