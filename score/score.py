import numpy as np
import torch
from tqdm import trange, tqdm
import math
import torch.nn.functional as F
from score.inception_network import InceptionV3
from score.fid_score import calculate_frechet_distance
from score.inception_score import kl_scores
#from .inception import InceptionV3
#from .fid_score import calculate_frechet_distance


def get_inception_and_fid_score(images, device, fid_cache, is_splits=10,
                                batch_size=50, verbose=False):
    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx1, block_idx2]).to(device)
    model.eval()

    if batch_size > len(images):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(images)

    fid_acts = np.empty((len(images), 2048))
    is_probs = np.empty((len(images), 1008))

    if verbose:
        iterator = trange(0, len(images), batch_size)
    else:
        iterator = range(0, len(images), batch_size)

    for start in iterator:
        end = start + batch_size
        batch_images = images[start: end]

        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)
        fid_acts[start: end] = pred[0].view(-1, 2048).cpu().numpy()
        is_probs[start: end] = pred[1].cpu().numpy()

    # Inception Score
    scores = []
    for i in range(is_splits):
        part = is_probs[
            (i * is_probs.shape[0] // is_splits):
            ((i + 1) * is_probs.shape[0] // is_splits), :]
        kl = part * (
            np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    is_score = (np.mean(scores), np.std(scores))

    # FID Score
    m1 = np.mean(fid_acts, axis=0)
    s1 = np.cov(fid_acts, rowvar=False)
    f = np.load(fid_cache)
    m2, s2 = f['mu'][:], f['sigma'][:]
    f.close()
    fid_score = calculate_frechet_distance(m1, s1, m2, s2)

    return is_score, fid_score

def compute_is_and_fid(gen, d, args, device, fid_cache, n_generate,splits):
    total_instance = n_generate
    n_batches = math.ceil(float(total_instance) / float(args.batch_size))
    ys = []
    pred_arr = np.empty((total_instance, 2048))
    # block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    # block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    inception_model = InceptionV3().to(device)
    inception_model.eval()
    for i in tqdm(range(0, n_batches)):
        start = i * args.batch_size
        end = start + args.batch_size
        z_train = torch.randn((args.batch_size, d)).cuda()
        batch_images = gen(z_train)
        batch_images = batch_images.to(device)

        with torch.no_grad():
            embeddings, logits = inception_model(batch_images)
            y = torch.nn.functional.softmax(logits, dim=1)
        ys.append(y)

        if total_instance >= args.batch_size:
            pred_arr[start:end] = embeddings.cpu().data.numpy().reshape(args.batch_size, -1)
        else:
            pred_arr[start:] = embeddings[:total_instance].cpu().data.numpy().reshape(total_instance, -1)

        total_instance -= batch_images.shape[0]
    with torch.no_grad():
        ys = torch.cat(ys, 0)
    is_scores, is_std = kl_scores(ys[:n_generate], splits=splits)
    m1 = np.mean(pred_arr, axis=0)
    s1 = np.cov(pred_arr, rowvar=False)
    # total_instance = len(data_loader.dataset)
    # n_batches = math.ceil(float(total_instance) / float(args.batch_size))
    # data_iter = iter(data_loader)
    #
    # for i in tqdm(range(0, n_batches)):
    #     feed_list = next(data_iter)
    #     images = feed_list[0]
    #     images = images.to(device)
    #     start = i * args.batch_size
    #     end = start + args.batch_size
    #     with torch.no_grad():
    #         embeddings, logits = inception_model(images)
    #
    #     if total_instance >= args.batch_size:
    #         pred_arr[start:end] = embeddings.cpu().data.numpy().reshape(args.batch_size, -1)
    #     else:
    #         pred_arr[start:] = embeddings[:total_instance].cpu().data.numpy().reshape(total_instance, -1)
    #     total_instance -= images.shape[0]
    #m2 = np.mean(pred_arr, axis=0)
    #s2 = np.cov(pred_arr, rowvar=False)
    f = np.load(fid_cache)
    m2, s2 = f['mu'][:], f['sigma'][:]
    f.close()
    fid_score = calculate_frechet_distance(m1, s1, m2, s2)
    return is_scores,fid_score