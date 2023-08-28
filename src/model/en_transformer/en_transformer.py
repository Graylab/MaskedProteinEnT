'''
En-Transformer implementation is 
based on (almost identical to) the github repository from
EnTransformer (Attention adaptation of EGNN)
https://github.com/lucidrains/En-transformer


Main edits include modification for multiatom inputs:
Changing rotary embeddings to be based on residue positions
Other changes will be incorported if need arises
'''


import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint_sequential

from en_transformer.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def default(val, d):
    return val if exists(val) else d

def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# classes

# this follows the same strategy for normalization as done in SE3 Transformers
# https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/se3_transformer_pytorch.py#L95

class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-8, scale_init = 1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.attn_dict = None

    def forward(self, feats, coors, return_attn=False, **kwargs):
        feats = self.norm(feats)
        feats, coors = self.fn(feats, coors, **kwargs)
        if return_attn:
            self.attn_dict = self.fn.attn_dict
            #print('prenorm:',self.attn_dict.keys())
        return feats, coors

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, feats, coors, return_attn=False, **kwargs):
        feats_out, coors_delta = self.fn(feats, coors, return_attn=return_attn, **kwargs)
        if return_attn:
            self.attn_dict = self.fn.attn_dict
            #print('residual', self.attn_dict.keys())
        return feats + feats_out, coors + coors_delta

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4 * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, feats, coors):
        return self.net(feats), 0

class EquivariantAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 4,
        edge_dim = 0,
        coors_hidden_dim = 16,
        neighbors = 0,
        only_sparse_neighbors = False,
        valid_neighbor_radius = float('inf'),
        init_eps = 1e-3,
        rel_pos_emb = None,
        edge_mlp_mult = 2,
        norm_rel_coors = True,
        norm_coors_scale_init = 1.,
        use_cross_product = False,
        talking_heads = False,
        rotary_theta = 10000,
        rel_dist_cutoff = 5000,
        rel_dist_scale = 1e2,
        dropout = 0.,
        num_atoms=1,
        edges_values=False,
        edges_from_nodes=False

    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        self.neighbors = neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_neighbor_radius = valid_neighbor_radius
        self.num_atoms = num_atoms

        attn_inner_dim = heads * dim_head
        self.heads = heads
        self.to_qkv = nn.Linear(dim, attn_inner_dim * 3, bias = False)
        self.to_out = nn.Linear(attn_inner_dim, dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else None

        self.edge_mlp = None
        self.edges_values = edges_values
        has_edges = edge_dim > 0
        self.edges_from_nodes = edges_from_nodes

        if has_edges or self.edges_from_nodes:
            edge_input_dim = heads + edge_dim
            edge_hidden = edge_input_dim * edge_mlp_mult

            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_input_dim, edge_hidden),
                nn.GELU(),
                nn.Linear(edge_hidden, heads)
            )

            self.coors_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(heads, heads)
            )
        else:
            self.coors_mlp = nn.Sequential(
                nn.Linear(heads, coors_hidden_dim),
                nn.GELU(),
                nn.Linear(coors_hidden_dim, heads)
            )

        self.coors_gate = nn.Sequential(
            nn.Linear(heads, heads),
            nn.Tanh()
        )

        self.use_cross_product = use_cross_product
        if use_cross_product:
            self.cross_coors_mlp = nn.Sequential(
                nn.Linear(heads, coors_hidden_dim),
                nn.GELU(),
                nn.Linear(coors_hidden_dim, heads * 2)
            )

        self.norm_rel_coors = CoorsNorm(scale_init = norm_coors_scale_init) if norm_rel_coors else nn.Identity()

        num_coors_combine_heads = (2 if use_cross_product else 1) * heads
        self.coors_combine = nn.Parameter(torch.randn(num_coors_combine_heads))

        self.rotary_emb = SinusoidalEmbeddings(dim_head // (2 if rel_pos_emb else 1), theta = rotary_theta)
        self.rotary_emb_seq = SinusoidalEmbeddings(dim_head // 2, theta = rotary_theta) if rel_pos_emb else None

        self.rel_dist_cutoff = rel_dist_cutoff
        self.rel_dist_scale = rel_dist_scale

        #self.node_dropout = nn.Dropout(dropout)
        #self.coor_dropout = nn.Dropout(dropout)

        self.init_eps = init_eps
        self.apply(self.init_)
        self.attn_dict = None

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.normal_(module.weight, std = self.init_eps)

    def forward(
        self,
        feats,
        coors,
        edges = None,
        mask = None,
        adj_mat = None,
        pos_indices = None
    ):
        b, n, d, h, num_nn, only_sparse_neighbors, valid_neighbor_radius, device = *feats.shape, self.heads, self.neighbors, self.only_sparse_neighbors, self.valid_neighbor_radius, feats.device

        assert not (only_sparse_neighbors and not exists(adj_mat)), 'adjacency matrix must be passed in if only_sparse_neighbors is turned on'

        if exists(mask):
            num_nodes = mask.sum(dim = -1)

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = rel_coors.norm(p = 2, dim = -1)

        # calculate neighborhood indices

        nbhd_indices = None
        nbhd_masks = None
        nbhd_ranking = rel_dist.clone()

        if exists(adj_mat):
            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat, 'i j -> b i j', b = b)

            self_mask = torch.eye(n, device = device).bool()
            self_mask = rearrange(self_mask, 'i j -> () i j')
            adj_mat.masked_fill_(self_mask, False)

            max_adj_neighbors = adj_mat.long().sum(dim = -1).max().item() + 1

            num_nn = max_adj_neighbors if only_sparse_neighbors else (num_nn + max_adj_neighbors)
            valid_neighbor_radius = 0 if only_sparse_neighbors else valid_neighbor_radius

            nbhd_ranking = nbhd_ranking.masked_fill(self_mask, -1.)
            nbhd_ranking = nbhd_ranking.masked_fill(adj_mat, 0.)

        if 0 < num_nn < n:
            # make sure padding does not end up becoming neighbors
            if exists(mask):
                ranking_mask = mask[:, :, None] * mask[:, None, :]
                nbhd_ranking = nbhd_ranking.masked_fill(~ranking_mask, 1e5)

            nbhd_values, nbhd_indices = nbhd_ranking.topk(num_nn, dim = -1, largest = False)
            nbhd_masks = nbhd_values <= valid_neighbor_radius

        # derive queries keys and values

        q, k, v = self.to_qkv(feats).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # calculate nearest neighbors

        i = j = n

        self.attn_dict = {}
        if exists(nbhd_indices):
            i, j = nbhd_indices.shape[-2:]
            nbhd_indices_with_heads = repeat(nbhd_indices, 'b n d -> b h n d', h = h)
            k         = batched_index_select(k, nbhd_indices_with_heads, dim = 2)
            v         = batched_index_select(v, nbhd_indices_with_heads, dim = 2)
            rel_dist  = batched_index_select(rel_dist, nbhd_indices, dim = 2)
            rel_coors = batched_index_select(rel_coors, nbhd_indices, dim = 2)
            self.attn_dict['nbhd_mask'] = nbhd_indices_with_heads
        else:
            k = repeat(k, 'b h j d -> b h n j d', n = n)
            v = repeat(v, 'b h j d -> b h n j d', n = n)

        # prepare mask

        if exists(mask):
            q_mask = rearrange(mask, 'b i -> b () i ()')
            k_mask = repeat(mask, 'b j -> b i j', i = n)

            if exists(nbhd_indices):
                k_mask = batched_index_select(k_mask, nbhd_indices, dim = 2)

            k_mask = rearrange(k_mask, 'b i j -> b () i j')

            mask = q_mask * k_mask

            if exists(nbhd_masks):
                mask &= rearrange(nbhd_masks, 'b i j -> b () i j')

        # generate and apply rotary embeddings

        q_pos_emb_rel_dist = self.rotary_emb(torch.zeros(n, device = device))

        rel_dist_to_rotate = (rel_dist * self.rel_dist_scale).clamp(max = self.rel_dist_cutoff)
        k_pos_emb_rel_dist = self.rotary_emb(rel_dist_to_rotate)

        q_pos_emb = rearrange(q_pos_emb_rel_dist, 'i d -> () () i d')
        k_pos_emb = rearrange(k_pos_emb_rel_dist, 'b i j d -> b () i j d')

        if exists(self.rotary_emb_seq):
            if pos_indices is not None:
                assert pos_indices.shape[0] == n
            elif self.num_atoms == 1:
                pos_indices = torch.arange(n, device = device)
            else:
                pos_indices_res = torch.arange(n // self.num_atoms, device = device)
                pos_indices = torch.repeat_interleave(pos_indices_res, self.num_atoms)
                assert pos_indices.shape[0] == n
            pos_emb = self.rotary_emb_seq(pos_indices)

            q_pos_emb_seq = rearrange(pos_emb, 'n d -> () () n d')
            k_pos_emb_seq = rearrange(pos_emb, 'n d -> () () n () d')

            q_pos_emb = broadcat((q_pos_emb, q_pos_emb_seq), dim = -1)
            k_pos_emb = broadcat((k_pos_emb, k_pos_emb_seq), dim = -1)

        q = apply_rotary_pos_emb(q, q_pos_emb)
        k = apply_rotary_pos_emb(k, k_pos_emb)
        v = apply_rotary_pos_emb(v, k_pos_emb)

        # calculate inner product for queries and keys

        qk = einsum('b h i d, b h i j d -> b h i j', q, k) * (self.scale if not exists(edges) else 1)

        # add edge information and pass through edges MLP if needed
        #if self.edges_from_nodes:
            # Same as Egnn in Ablooper
            
            #feats_i = rearrange(feats, 'b i d -> b i () d')
            #print(feats_i.shape)
            #if exists(nbhd_indices):
                #print(feats.shape)
            #    feats_j = batched_index_select(feats, nbhd_indices, dim=1)
                #print(feats_j.shape)
            #    feats_i = feats_i.expand(-1, -1, feats_j.shape[2],-1)
                #print(feats_i.shape)
            #else:
            #    feats_j = rearrange(feats, 'b j d -> b () j d')
                #print('all ', feats_j.shape)
                
            #feats_i, feats_j = torch.broadcast_tensors(feats_i, feats_j)
            #print(rel_dist.shape)
            #edges = torch.cat((feats_i, feats_j, rel_dist.unsqueeze(-1)), dim=-1)
            #print('adding edges: ', edges.shape)

        if exists(edges):
            if exists(nbhd_indices):
                edges = batched_index_select(edges, nbhd_indices, dim = 2)

            qk = rearrange(qk, 'b h i j -> b i j h')
            qk = torch.cat((qk, edges), dim = -1)
            qk = self.edge_mlp(qk)
            qk = rearrange(qk, 'b i j h -> b h i j')
        
        # coordinate MLP and calculate coordinate updates

        coors_mlp_input = rearrange(qk, 'b h i j -> b i j h')
        coor_weights = self.coors_mlp(coors_mlp_input)

        if exists(mask):
            mask_value = max_neg_value(coor_weights)
            coor_mask = repeat(mask, 'b () i j -> b i j ()')
            coor_weights.masked_fill_(~coor_mask, mask_value)

        #coor_weights = coor_weights - coor_weights.amax(dim = -2, keepdim = True).detach()
        coor_attn = coor_weights.softmax(dim = -2)
        #coor_attn = self.coor_dropout(coor_attn)

        rel_coors_sign = self.coors_gate(coors_mlp_input)
        rel_coors_sign = rearrange(rel_coors_sign, 'b i j h -> b i j () h')

        if self.use_cross_product:
            rel_coors_i = repeat(rel_coors, 'b n i c -> b n (i j) c', j = j)
            rel_coors_j = repeat(rel_coors, 'b n j c -> b n (i j) c', i = j)

            cross_coors = torch.cross(rel_coors_i, rel_coors_j, dim = -1)

            cross_coors = self.norm_rel_coors(cross_coors)
            cross_coors = repeat(cross_coors, 'b i j c -> b i j c h', h = h)

        rel_coors = self.norm_rel_coors(rel_coors)
        rel_coors = repeat(rel_coors, 'b i j c -> b i j c h', h = h)

        rel_coors = rel_coors * rel_coors_sign

        # cross product

        if self.use_cross_product:
            cross_weights = self.cross_coors_mlp(coors_mlp_input)

            cross_weights = rearrange(cross_weights, 'b i j (h n) -> b i j h n', n = 2)
            cross_weights_i, cross_weights_j = cross_weights.unbind(dim = -1)

            cross_weights = rearrange(cross_weights_i, 'b n i h -> b n i () h') + rearrange(cross_weights_j, 'b n j h -> b n () j h')

            if exists(mask):
                cross_mask = (coor_mask[:, :, :, None, :] & coor_mask[:, :, None, :, :])
                cross_weights = cross_weights.masked_fill(~cross_mask, mask_value)

            cross_weights = rearrange(cross_weights, 'b n i j h -> b n (i j) h')
            cross_attn = cross_weights.softmax(dim = -2)

        # aggregate and combine heads for coordinate updates

        rel_out = einsum('b i j h, b i j c h -> b i c h', coor_attn, rel_coors)

        if self.use_cross_product:
            cross_out = einsum('b i j h, b i j c h -> b i c h', cross_attn, cross_coors)
            rel_out = torch.cat((rel_out, cross_out), dim = -1)

        coors_out = einsum('b n c h, h -> b n c', rel_out, self.coors_combine)

        # derive attention

        sim = qk.clone()

        if exists(mask):
            mask_value = max_neg_value(sim)
            sim.masked_fill_(~mask, mask_value)

        #sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        #attn = self.node_dropout(attn)
        self.attn_dict['attn'] = attn

        if exists(self.talking_heads):
            attn = self.talking_heads(attn)
            self.attn_dict['th_attn'] = attn
        
        #print('attn', self.attn_dict.keys())
        #print('attn', self.attn_dict['attn'].shape)

        # weighted sum of values and combine heads

        out = einsum('b h i j, b h i j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, coors_out

# transformer

class Block(nn.Module):
    def __init__(self, attn, ff):
        super().__init__()
        self.attn = attn
        self.ff = ff

    def forward(self, inp, return_attn=False):
        feats, coors, mask, edges, adj_mat, pos_indices = inp
        feats, coors = self.attn(feats, coors, edges = edges, mask = mask, 
                                    adj_mat = adj_mat, pos_indices=pos_indices,
                                    return_attn=return_attn)
        feats, coors = self.ff(feats, coors)
        if return_attn:
            return (feats, coors, mask, edges, adj_mat, pos_indices, self.attn.attn_dict)    
        return (feats, coors, mask, edges, adj_mat, pos_indices)

class EnTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_tokens = None,
        rel_pos_emb = False,
        dim_head = 64,
        heads = 8,
        num_edge_tokens = None,
        edge_dim = 0,
        coors_hidden_dim = 16,
        neighbors = 0,
        only_sparse_neighbors = False,
        num_adj_degrees = None,
        adj_dim = 0,
        valid_neighbor_radius = float('inf'),
        init_eps = 1e-3,
        norm_rel_coors = True,
        norm_coors_scale_init = 1.,
        use_cross_product = False,
        talking_heads = False,
        checkpoint = False,
        n_checkpoints = 1,
        rotary_theta = 10000,
        rel_dist_cutoff = 5000,
        rel_dist_scale = 1e2,
        attn_dropout = 0.,
        ff_dropout = 0.,
        num_atoms = 1,
        edges_values=False,
        edges_from_nodes=False
    ):
        super().__init__()
        assert dim_head >= 32, 'your dimension per head should be greater than 32 for rotary embeddings to work well'
        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than 1'

        if only_sparse_neighbors:
            num_adj_degrees = default(num_adj_degrees, 1)

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None
        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None

        self.num_adj_degrees = num_adj_degrees
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None
        adj_dim = adj_dim if exists(num_adj_degrees) else 0

        self.checkpoint = checkpoint
        self.n_checkpoints = n_checkpoints
        self.layers = nn.ModuleList([])

        for ind in range(depth):
            self.layers.append(Block(
                Residual(PreNorm(dim, EquivariantAttention(dim = dim, dim_head = dim_head, \
                    heads = heads, coors_hidden_dim = coors_hidden_dim, edge_dim = (edge_dim + adj_dim),
                     neighbors = neighbors, only_sparse_neighbors = only_sparse_neighbors, \
                         valid_neighbor_radius = valid_neighbor_radius, init_eps = init_eps, \
                             rel_pos_emb = rel_pos_emb, norm_rel_coors = norm_rel_coors, \
                                 norm_coors_scale_init = norm_coors_scale_init, 
                                 use_cross_product = use_cross_product, 
                                 talking_heads = talking_heads, 
                                 rotary_theta = rotary_theta, 
                                 rel_dist_cutoff = rel_dist_cutoff, 
                                 rel_dist_scale = rel_dist_scale, 
                                 dropout = attn_dropout,
                                 num_atoms=num_atoms,
                                 edges_values=edges_values,
                                 edges_from_nodes=edges_from_nodes))),
                Residual(PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout)))
            ))

    def forward(
        self,
        feats,
        coors,
        edges = None,
        mask = None,
        adj_mat = None,
        return_coor_changes = False,
        pos_indices = None,
        return_attn=False,
        **kwargs
    ):
        b = feats.shape[0]

        if exists(self.token_emb):
            feats = self.token_emb(feats)

        if exists(self.edge_emb):
            assert exists(edges), 'edges must be passed in as (batch x seq x seq) indicating edge type'
            edges = self.edge_emb(edges)

        assert not (exists(adj_mat) and (not exists(self.num_adj_degrees) or self.num_adj_degrees == 0)), 'num_adj_degrees must be greater than 0 if you are passing in an adjacency matrix'

        if exists(self.num_adj_degrees):
            assert exists(adj_mat), 'adjacency matrix must be passed in (keyword argument adj_mat)'

            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

            adj_indices = adj_mat.clone().long()

            for ind in range(self.num_adj_degrees - 1):
                degree = ind + 2

                next_degree_adj_mat = (adj_mat.float() @ adj_mat.float()) > 0
                next_degree_mask = (next_degree_adj_mat.float() - adj_mat.float()).bool()
                adj_indices.masked_fill_(next_degree_mask, degree)
                adj_mat = next_degree_adj_mat.clone()

            if exists(self.adj_emb):
                adj_emb = self.adj_emb(adj_indices)
                edges = torch.cat((edges, adj_emb), dim = -1) if exists(edges) else adj_emb

        assert not (return_coor_changes and self.training), 'you must be eval mode in order to return coordinates'

        # go through layers

        coor_changes = [coors]
        per_layer_rep = []
        inp = (feats, coors, mask, edges, adj_mat, pos_indices)
        
        # if in training mode and checkpointing is designated, use checkpointing across blocks to save memory
        # TODO: Figure out why peformance drops on 
        if self.training and self.checkpoint:
            inp[0].requires_grad_()
            inp = checkpoint_sequential(self.layers, self.n_checkpoints, inp)
        else:
            # iterate through blocks
            attn_dict_per_layer = {}
            for layer in self.layers:
                if return_attn:
                    inp_and_attn = layer(inp, return_attn=return_attn)
                    #print(len(inp_and_attn))
                    inp = (inp_and_attn[:-1])
                    per_layer_rep.append(inp[0])
                    attn_dict = inp_and_attn[-1]
                    for key in attn_dict:
                        if key in attn_dict_per_layer:
                            attn_dict_per_layer[key].append(attn_dict[key])
                        else:
                            attn_dict_per_layer[key] = [attn_dict[key]]
                else:
                    inp = layer(inp)
                coor_changes.append(inp[1]) # append coordinates for visualization

        # return

        feats, coors, *_ = inp

        if return_coor_changes:
            return feats, coors, coor_changes
        
        if return_attn:
            attn_dict_per_layer['rep'] = per_layer_rep
            return feats, coors, attn_dict_per_layer

        return feats, coors
