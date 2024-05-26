import json
import os
import subprocess
import tempfile
import time
import gzip
import pickle
import numpy as np
import torch
import torch.distributed as dist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# from mpi4py import MPI

NUM_GPUS = int(os.environ["WORLD_SIZE"])
print('NUM_GPUS:', NUM_GPUS)


def allreduce(x, average):
    if mpi_size() > 1:
        dist.all_reduce(x, dist.ReduceOp.SUM)
    return x / mpi_size() if average else x


def get_cpu_stats_over_ranks(stat_dict):
    keys = sorted(stat_dict.keys())
    allreduced = allreduce(torch.stack(
        [torch.as_tensor(stat_dict[k]).detach().cuda().float() for k in keys]),
                           average=True).cpu()
    return {k: allreduced[i].item() for (i, k) in enumerate(keys)}


class Hyperparams(dict):

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


def logger(log_prefix):
    'Prints the arguments out to stdout, .txt, and .jsonl files'

    jsonl_path = f'{log_prefix}.jsonl'
    txt_path = f'{log_prefix}.txt'

    def log(*args, pprint=False, **kwargs):
        if mpi_rank() != 0:
            return
        t = time.ctime()
        argdict = {'time': t}
        if len(args) > 0:
            argdict['message'] = ' '.join([str(x) for x in args])
        argdict.update(kwargs)

        txt_str = []
        args_iter = sorted(argdict) if pprint else argdict
        for k in args_iter:
            val = argdict[k]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            elif isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)
            argdict[k] = val
            if isinstance(val, float):
                val = f'{val:.5f}'
            txt_str.append(f'{k}: {val}')
        txt_str = ', '.join(txt_str)

        if pprint:
            json_str = json.dumps(argdict, sort_keys=True)
            txt_str = json.dumps(argdict, sort_keys=True, indent=4)
        else:
            json_str = json.dumps(argdict)

        print(txt_str, flush=True)

        with open(txt_path, "a+") as f:
            print(txt_str, file=f, flush=True)
        with open(jsonl_path, "a+") as f:
            print(json_str, file=f, flush=True)

    return log


def maybe_download(path, filename=None):
    '''If a path is a gsutil path, download it and return the local link,
    otherwise return link'''
    if not path.startswith('gs://'):
        return path
    if filename:
        local_dest = f'/tmp/'
        out_path = f'/tmp/{filename}'
        if os.path.isfile(out_path):
            return out_path
        subprocess.check_output(['gsutil', '-m', 'cp', '-R', path, out_path])
        return out_path
    else:
        local_dest = tempfile.mkstemp()[1]
        subprocess.check_output(['gsutil', '-m', 'cp', path, local_dest])
    return local_dest


def tile_images(images, d1=4, d2=4, border=1):
    id1, id2, c = images[0].shape
    out = np.ones(
        [d1 * id1 + border * (d1 + 1), d2 * id2 + border * (d2 + 1), c],
        dtype=np.uint8)
    out *= 255
    if len(images) != d1 * d2:
        raise ValueError('Wrong num of images')
    for imgnum, im in enumerate(images):
        num_d1 = imgnum // d2
        num_d2 = imgnum % d2
        start_d1 = num_d1 * id1 + border * (num_d1 + 1)
        start_d2 = num_d2 * id2 + border * (num_d2 + 1)
        out[start_d1:start_d1 + id1, start_d2:start_d2 + id2, :] = im
    return out


def image_grid(img, grid_h, grid_w, color=(255, 255, 255)):

    img_h, img_w, _ = img.shape

    for idx_h in range(1, img_h // grid_h):
        for idx_w in range(1, img_w // grid_w):
            border_h = idx_h * grid_h
            border_w = idx_w * grid_w
            img[border_h, :, :3] = color
            if idx_w % 2 == 0:
                margin = 1
            else:
                margin = 0
            img[:, border_w - margin:border_w + margin + 1, :3] = color

    return img


def arrange_side_by_side(array_a: np.array, array_b: np.array) -> np.array:
    """
    Takes two numpy arrays and arranges their elements side-by-side along the
    first dimension.

    Ex:
        If array_a is [1, 2, 3] and array_b is [4, 5, 6], the output will be
        [1, 4, 2, 5, 3, 6].
    """
    a = []
    num_arrays = array_a.shape[0]
    for idx in range(num_arrays):
        elem_a = array_a[idx]
        elem_b = array_b[idx]
        a.append(elem_a)
        a.append(elem_b)
    a = np.stack(a)

    return a


def mpi_size():
    return int(os.environ['WORLD_SIZE'])  # MPI.COMM_WORLD.Get_size()


def mpi_rank():
    return int(os.environ['RANK'])  # MPI.COMM_WORLD.Get_rank()


def num_nodes():
    nn = mpi_size()
    if nn % NUM_GPUS == 0:
        return nn // NUM_GPUS
    return nn // NUM_GPUS + 1


def gpus_per_node():
    size = mpi_size()
    if size > 1:
        return max(size // num_nodes(), 1)
    return 1


def local_mpi_rank():
    return mpi_rank() % gpus_per_node()


def apply_pca_to_embmap(embmap, pca=None, n_components=3):
    '''
    Args:
        embmap: (H, W, C) np.float32 array.
        pca: Pre-fitted PCA object. Fits new PCA object if not provided.
        n_components: Number of components to keep.
    '''
    # Reshape the embedding map to 2D
    embmap_reshaped = embmap.reshape(-1, embmap.shape[-1])

    # Remove empty elements from PCA projection computation
    mask = np.all(embmap_reshaped == 0, axis=-1)
    mask = np.logical_not(mask)
    embmap_reshaped_nonempty = embmap_reshaped[mask]

    # Compute PCA
    if pca is None:
        pca = PCA(n_components=n_components)
        embmap_pca = pca.fit(embmap_reshaped_nonempty)

    # Apply PCA
    embmap_pca = pca.transform(embmap_reshaped)

    # Reshape the PCA-transformed data back to the original shape
    embmap_pca = embmap_pca.reshape(embmap.shape[0], embmap.shape[1],
                                    n_components)

    # Scale the values in embmap_pca to the range [0, 1]
    scaler = MinMaxScaler()
    embmap_pca = scaler.fit_transform(embmap_pca.reshape(
        -1, n_components)).reshape(embmap.shape[0], embmap.shape[1], n_components)

    # Clip the values in embmap_pca to the range [0, 1]
    embmap_pca = np.clip(embmap_pca, 0, 1)

    return embmap_pca, pca


def embmap2rgb_pca(embmap: torch.tensor, pca=None) -> tuple:
    '''
    Transforms a batched unit vector embedding map to an RGB image using PCA.

    Args:
        embmap: Unit vector embedding map (B, H, W, D).
        pca: Pre-fitted PCA object. Fits new PCA object if not provided.

    Returns:
        embmap_rgb (torch.tensor): RGB image (B, H, W, 3).
        pca: Fitted PCA object.
    '''
    embmap = embmap.cpu().numpy()

    # Create RGB sample-by-sample
    B = embmap.shape[0]
    x_rgbs = []
    for batch_idx in range(B):
        x = embmap[batch_idx]
        x_pca, pca = apply_pca_to_embmap(x, pca, n_components=3)
        x_rgb = torch.tensor(x_pca*255, dtype=torch.uint8)
        # Replace zero vector elements with white color
        empty_mask = np.all(x == 0, axis=-1)
        x_rgb[empty_mask] = 255
        x_rgbs.append(x_rgb)
    
    embmap_rgb = torch.stack(x_rgbs)    

    return embmap_rgb, pca


def viz_rows2img(batches):
    '''
    Creates a concatenated image out of lists of visualization row images.

    Args:
        batches: List of (B, H, W, 3) np.uint8 arrays representing rows to viz.
    
    Returns:
        (N*H, B*W, 3) np.uint8 array of the concatenated viz rows.
    '''
    # Concatenate along the first axis
    concatenated = np.concatenate(batches, axis=0)

    # Get the shape of a single image
    B, H, W, _ = batches[0].shape
    N = len(batches)

    # Reshape to the desired shape
    img = concatenated.reshape(N, B, H, W, 3).transpose(0, 2, 1, 3, 4).reshape(N*H, B*W, 3)

    return img


def read_compressed_pickle(path):
    try:
        with gzip.open(path, "rb") as f:
            pkl_obj = f.read()
            obj = pickle.loads(pkl_obj)
            return obj
    except IOError as error:
        print(error)


def load_register(txt2idx_star_pth: str):
    '''Returns the specified txt_star2_idx_str register dict object from disk,
    or create an empty dict object if the file does not exist.
    '''
    if not os.path.isfile(txt2idx_star_pth):
        txt2_idx_str = {}
        return txt2_idx_str

    with open(txt2idx_star_pth, 'rb') as f:
        txt2_idx_str = pickle.load(f)

    return txt2_idx_str


def sample_inference(sample_pth:str, vae, dataset) -> torch.Tensor:
    '''
    Runs inference on sample and returns a (H, W, D) tensor in CPU.
    Args:
        sample_pth: Path to a compressed sample .pkl.gz file.
    '''
    sample = dataset.preproc_sample(sample_pth)  # (2, H, W, D)

    x_post_match = sample[0]
    # x_oracle = sample[1]

    x_post_match = torch.tensor(x_post_match).unsqueeze(0).cuda()

    with torch.no_grad():
        px_z = vae.inference(x_post_match)
    px_z = px_z[0].cpu()

    return px_z