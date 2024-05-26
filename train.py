import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import set_up_data

# from debug import viz_vae_forward_input
from train_helpers import (
    accumulate_stats,
    load_opt,
    load_vaes,
    save_model,
    set_up_hyperparams,
    update_ema,
)
from utils import get_cpu_stats_over_ranks
from viz import write_images


def training_step(H, data_input, target, vae, ema_vae, optimizer, iterate):
    '''
    Both input and targets need to be in the interval (-1, 1).
    Intensity that is blank is 0 for both oracle and posterior matching.

    Args:
        data_input: (B, 2, H, W, D) open vocab D dimensional unit vectors

    '''
    t0 = time.time()
    vae.zero_grad()

    # x_post_match: Partial
    # x_oracle:     Completed
    x_post_match = data_input[:, 0]
    x_oracle = data_input[:, 1]

    # Mask of observed target elements
    m_target = ~(torch.norm(x_oracle, dim=-1) == 0)

    stats = vae.forward(x_oracle, x_post_match, m_target)

    stats['elbo'].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(),
                                               H.grad_clip).item()
    distortion_nans = torch.isnan(stats['distortion']).sum()
    rate_nans = torch.isnan(stats['rate']).sum()
    stats.update(
        dict(rate_nans=0 if rate_nans == 0 else 1,
             distortion_nans=0 if distortion_nans == 0 else 1))
    stats = get_cpu_stats_over_ranks(stats)

    skipped_updates = 1
    # only update if no rank has a nan and if the grad norm is below a specific
    # threshold
    if stats['distortion_nans'] == 0 and stats['rate_nans'] == 0 and (
            H.skip_threshold == -1 or grad_norm < H.skip_threshold):
        optimizer.step()
        skipped_updates = 0
        update_ema(vae, ema_vae, H.ema_rate)

    t1 = time.time()
    stats.update(skipped_updates=skipped_updates,
                 iter_time=t1 - t0,
                 grad_norm=grad_norm)
    return stats


def eval_step(data_input, target, ema_vae):
    with torch.no_grad():
        # x_post_match: Partial
        # x_oracle:     Completed
        x_post_match = data_input[:, 0]
        x_oracle = data_input[:, 1]

        # Mask of observed target elements
        m_target = ~(torch.norm(x_oracle, dim=-1) == 0)

        stats = ema_vae.forward(x_oracle, x_post_match, m_target)

    stats = get_cpu_stats_over_ranks(stats)
    return stats


def get_sample_for_visualization(data, num):
    '''
    data --> (B, 2, H, W, D) tensors
    x: (B, H, W, D)

    Returns
        orig_image: RGB (np.uint8) image w. dim (B, H, W, C).
        preprocessed: Tensor in range (-1, 1) w. dim (B, H, W, C).
    '''
    for data_input in DataLoader(data, batch_size=num):
        break
    x = data_input[:, 0]  # (B, H, W, D)
    x_oracle = data_input[:, 1]  # (B, H, W, D)

    x = x.cuda(non_blocking=True).float()
    x_oracle = x_oracle.cuda(non_blocking=True).float()

    return x, x_oracle


def train_loop(H, data_train, data_valid, preprocess_fn, vae, ema_vae,
               logprint):
    optimizer, scheduler, cur_eval_loss, iterate, starting_epoch = load_opt(
        H, vae, logprint)
    train_sampler = DistributedSampler(data_train,
                                       num_replicas=H.mpi_size,
                                       rank=H.rank)

    num_viz = H.num_images_visualize
    x_viz, x_oracle_viz = get_sample_for_visualization(data_valid, num_viz)

    # Skip early evaluations to save time
    early_evals = set([9999999])  # set([1] + [2**exp for exp in range(3, 14)])
    stats = []
    iters_since_starting = 0
    H.ema_rate = torch.as_tensor(H.ema_rate).cuda()
    for epoch in range(starting_epoch, H.num_epochs):
        
        train_sampler.set_epoch(epoch)
        for x in DataLoader(data_train,
                            batch_size=H.n_batch,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=H.dataloader_workers,
                            sampler=train_sampler):
            # data_input, target = preprocess_fn(x)
            data_input, target = x, None
            training_stats = training_step(H, data_input, target, vae, ema_vae,
                                           optimizer, iterate)
            stats.append(training_stats)
            scheduler.step()
            if iterate % H.iters_per_print == 0 or iters_since_starting in early_evals:
                logprint(model=H.desc,
                         type='train_loss',
                         lr=scheduler.get_last_lr()[0],
                         epoch=epoch,
                         step=iterate,
                         **accumulate_stats(stats, H.iters_per_print))

            if iterate % H.iters_per_images == 0 or (
                    iters_since_starting in early_evals
                    and H.dataset != 'ffhq_1024') and H.rank == 0:

                for temp in H.viz_temps:
                    fname = f'{H.save_dir}/samples-{iterate}_t{temp}.png'
                    write_images(H,
                                 ema_vae,
                                 x_viz,
                                 x_oracle_viz,
                                 fname,
                                 logprint,
                                 temp=temp)

            iterate += 1
            iters_since_starting += 1
            if iterate % H.iters_per_save == 0 and H.rank == 0:
                if np.isfinite(stats[-1]['elbo']):
                    logprint(model=H.desc,
                             type='train_loss',
                             epoch=epoch,
                             step=iterate,
                             **accumulate_stats(stats, H.iters_per_print))
                    fp = os.path.join(H.save_dir, 'latest')
                    logprint(f'Saving model@ {iterate} to {fp}')
                    save_model(fp, vae, ema_vae, optimizer, H)

            if iterate % H.iters_per_ckpt == 0 and H.rank == 0:
                save_model(os.path.join(H.save_dir, f'iter-{iterate}'), vae,
                           ema_vae, optimizer, H)

        # if epoch % H.epochs_per_eval == 0:
        #     valid_stats = evaluate(H, ema_vae, data_valid, preprocess_fn)
        #     logprint(model=H.desc,
        #              type='eval_loss',
        #              epoch=epoch,
        #              step=iterate,
        #              **valid_stats)


def evaluate(H, ema_vae, data_valid, preprocess_fn):
    stats_valid = []
    valid_sampler = DistributedSampler(data_valid,
                                       num_replicas=H.mpi_size,
                                       rank=H.rank)
    for x in DataLoader(data_valid,
                        batch_size=H.n_batch,
                        drop_last=True,
                        pin_memory=True,
                        sampler=valid_sampler):
        x = x.cuda(non_blocking=True).float()
        data_input, target = x, None
        stats_valid.append(eval_step(data_input, target, ema_vae))

    if len(stats_valid) < 1:
        raise Exception('Evaluation output list is empty. '
                        f'Check sufficient val data size'
                        f'\n    len(valid_sampler) {len(valid_sampler)}'
                        f'\n    stats_valid {len(stats_valid)}')

    vals = [a['elbo'] for a in stats_valid]
    finites = np.array(vals)[np.isfinite(vals)]
    stats = dict(
        n_batches=len(vals),
        filtered_elbo=np.mean(finites),
        **{k: np.mean([a[k] for a in stats_valid])
           for k in stats_valid[-1]})
    return stats


def run_test_eval(H, ema_vae, data_test, preprocess_fn, logprint):
    print('evaluating')
    num_viz = H.num_images_visualize
    x_viz, x_oracle_viz = get_sample_for_visualization(data_test, num_viz)
    for temp in [0.1, 0.4, 1.0]:
        fname = f'{H.save_dir}/samples-eval_t{temp}.png'
        write_images(H, ema_vae, x_viz, x_oracle_viz, fname, logprint, temp)

    stats = evaluate(H, ema_vae, data_test, preprocess_fn)
    print('test results')
    for k in stats:
        print(k, stats[k])
    logprint(type='test_loss', **stats)


def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    vae, ema_vae = load_vaes(H, logprint)
    if H.test_eval:
        run_test_eval(H, ema_vae, data_valid_or_test, preprocess_fn, logprint)
    else:
        train_loop(H, data_train, data_valid_or_test, preprocess_fn, vae,
                   ema_vae, logprint)


if __name__ == "__main__":
    main()
