import torch
import numpy as np

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)


def get_cleanid_from_numpy_string(id):
    cleanid= str(id)[2:-1]
    return cleanid

def get_recovery_metrics_for_batch(batch, model, temp, N):
    import random
    random.seed(random.randrange(100000))
    if len(batch) == 3:
        id, (nfeats, coords, _, mask, labels, pos_indices), _ = batch
    else:
        id, (nfeats, coords, _, mask, labels, pos_indices) = batch
    labels = labels.cpu()
    nfeats_residues = nfeats.clone().reshape(nfeats.shape[0], labels.shape[0], -1,
                                             nfeats.shape[-1]
                                             )[:, :, 0, :20].squeeze(0)
    pred_indices = list(torch.nonzero(labels!=20).numpy().flatten())
    fixed_indices = list(torch.nonzero(labels==20).numpy().flatten())
    fixed_labels = nfeats_residues[fixed_indices, :].argmax(dim=-1)
    nfeats = nfeats.to(device) #one hot
    coords = coords.to(device)
    mask = mask.to(device)
    model = model.to(device)
    pos_indices = pos_indices.to(device)
    y_hat, _, _ = model(nfeats.double(), coords.double(), mask=mask, pos_indices=pos_indices)
    y_hat = y_hat.cpu()
    dim_ignore_index = torch.full((y_hat.shape[0], 1),
                            float('-inf')).type_as(y_hat)
    y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)
    #loss
    loss = torch.nn.functional.cross_entropy(y_hat,
                                            labels,
                                            ignore_index=20,
                                            reduction='mean')
    loss_full = torch.nn.functional.cross_entropy(y_hat,
                                            labels,
                                            ignore_index=20,
                                            reduction='none')
    #argmax
    y_argmax = y_hat_ignore_index.argmax(dim=-1)
    correct_unmasked_argmax = (y_argmax == labels).long()
    target_unmasked = torch.ones(labels.shape).type_as(labels)
    keep_tensor = (~(labels == 20)).type_as(labels)
    correct_argmax = torch.sum(correct_unmasked_argmax*keep_tensor)
    total = torch.sum(target_unmasked*keep_tensor)
    print('Total nodes ', total.item())
    indices = torch.nonzero(keep_tensor).flatten().tolist()
    if total == 0:
        total = 1.0
    seqrecargmax = correct_argmax / float(total)
    y_argmax[fixed_indices] = fixed_labels
    y_wt = labels
    y_wt[fixed_indices] = fixed_labels

    #perplexity
    cleanid = get_cleanid_from_numpy_string(id[0])
    # sampled
    seqrecsampled_all = []
    seqsampled_all = []
    for _ in range(N):
        y_predicted_all = torch.nn.functional.softmax(y_hat/temp, dim=1)
        y_sampled = torch.multinomial(y_predicted_all, 1, replacement=True).squeeze(1)
        correct_unmasked = (y_sampled == labels).long()
        correct = torch.sum(correct_unmasked*keep_tensor)   
        seqrecsampled_all.append(float(correct) / float(total))

        y_sampled[fixed_indices] = fixed_labels
        seqsampled_all.append(y_sampled.numpy())

    return {'id':cleanid,
            'seqrecsampled_all': seqrecsampled_all,
            'seqrecargmax': float(seqrecargmax),
            'loss': loss,
            'total_nodes': total,
            'sequence_argmax': y_argmax.numpy(),
            'sequences_sampled': seqsampled_all,
            'wt': y_wt.numpy(),
            'loss_full': loss_full.cpu().numpy(),
            'pred_indices': pred_indices,
            'fixed_indices': fixed_indices,
            'correct': correct_unmasked_argmax,
            'design_indices': indices
            }


def score_sequences(batch, labels, model):
    import random
    random.seed(random.randrange(100000))
    if len(batch) == 3:
        id, (nfeats, coords, _, mask, _, pos_indices), _ = batch
    else:
        id, (nfeats, coords, _, mask, _, pos_indices) = batch
    labels = labels.cpu()
    nfeats_residues = nfeats.clone().reshape(nfeats.shape[0], labels.shape[0], -1,
                                             nfeats.shape[-1]
                                             )[:, :, 0, :20].squeeze(0)
    pred_indices = list(torch.nonzero(labels!=20).numpy().flatten())
    fixed_indices = list(torch.nonzero(labels==20).numpy().flatten())
    fixed_labels = nfeats_residues[fixed_indices, :].argmax(dim=-1)
    nfeats = nfeats.to(device) #one hot
    coords = coords.to(device)
    mask = mask.to(device)
    model = model.to(device)
    pos_indices = pos_indices.to(device)
    y_hat, _, _ = model(nfeats.double(), coords.double(), mask=mask, pos_indices=pos_indices)
    y_hat = y_hat.cpu()
    dim_ignore_index = torch.full((y_hat.shape[0], 1),
                            float('-inf')).type_as(y_hat)
    y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)
    #loss
    score_labels = []
    for label_seq in labels:
        loss = torch.nn.functional.cross_entropy(y_hat,
                                                label_seq,
                                                ignore_index=20,
                                                reduction='mean')
        score_labels.append(loss.item())
    #argmax
    y_argmax = y_hat_ignore_index.argmax(dim=-1)
    correct_unmasked_argmax = (y_argmax == labels).long()
    target_unmasked = torch.ones(labels.shape).type_as(labels)
    keep_tensor = (~(labels == 20)).type_as(labels)
    correct_argmax = torch.sum(correct_unmasked_argmax*keep_tensor)
    total = torch.sum(target_unmasked*keep_tensor)
    indices = torch.nonzero(keep_tensor).flatten().tolist()
    if total == 0:
        total = 1.0
    seqrecargmax = correct_argmax / float(total)
    y_argmax[fixed_indices] = fixed_labels
    y_wt = labels
    y_wt[fixed_indices] = fixed_labels

    #perplexity
    cleanid = get_cleanid_from_numpy_string(id[0])
    # sampled
    
    return {'id':cleanid,
            'seqrecargmax': float(seqrecargmax),
            'scores': score_labels,
            'total_nodes': total,
            'sequence_argmax': y_argmax.numpy(),
            'wt': y_wt.numpy(),
            'pred_indices': pred_indices,
            'fixed_indices': fixed_indices,
            'correct': correct_unmasked_argmax
            }
