import numpy as np
import torch
from tqdm import trange

from .inception import InceptionV3


def get_inception_score(images, device, splits=10, batch_size=32,
                        verbose=False):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    preds = []
    if verbose:
        iterator = trange(0, len(images), batch_size, dynamic_ncols=True)
    else:
        iterator = range(0, len(images), batch_size)
    for start in iterator:
        end = start + batch_size
        batch_images = images[start: end]
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        pred = model(batch_images)[0]
        preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        part = preds[
            (i * preds.shape[0] // splits):
            ((i + 1) * preds.shape[0] // splits), :]
        kl = part * (
            np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def kl_scores(ys, splits):
    scores = []
    n_images = ys.shape[0]
    with torch.no_grad():
        for j in range(splits):
            part = ys[(j*n_images//splits): ((j+1)*n_images//splits), :]
            kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            kl = torch.exp(kl)
            scores.append(kl.unsqueeze(0))
        scores = torch.cat(scores, 0)
        m_scores = torch.mean(scores).detach().cpu().numpy()
        m_std = torch.std(scores).detach().cpu().numpy()
    return m_scores, m_std
