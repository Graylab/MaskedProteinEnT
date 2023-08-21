from typing import Optional
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy
from src.models.en_transformer.en_transformer import EnTransformer
from src.models.ProteinMaskedLabelModel.MaskedModelMetrics import LocalAccuracy
from src.models.ProteinMaskedLabelModel.Schedulers import CosineWarmupScheduler

'''
EnTransformer (Attention adaptation of EGNN)
https://github.com/lucidrains/En-transformer
Original: 
https://github.com/vgsatorras/egnn/blob/3c079e7267dad0aa6443813ac1a12425c3717558/models/egnn_clean/egnn_clean.py#L5
Citations:

@misc{satorras2021en,
    title 	= {E(n) Equivariant Graph Neural Networks}, 
    author 	= {Victor Garcia Satorras and Emiel Hoogeboom and Max Welling},
    year 	= {2021},
    eprint 	= {2102.09844},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
@misc{shazeer2020talkingheads,
    title   = {Talking-Heads Attention}, 
    author  = {Noam Shazeer and Zhenzhong Lan and Youlong Cheng and Nan Ding and Le Hou},
    year    = {2020},
    eprint  = {2003.02436},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
@misc{su2021roformer,
    title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
    author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
    year    = {2021},
    eprint  = {2104.09864},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
'''
torch.set_default_dtype(torch.float64)

class ProteinMaskedLabelModel_EnT_MA(pl.LightningModule):
    def __init__(self, **config):

        super().__init__()
        self.save_hyperparameters()
        self.n_hidden = config['n_hidden']
        self.max_seq_len = config['max_seq_len']
        self.layers = config['layers']
        self.n_heads = config['n_heads']
        self.dropout = config['dropout']
        self.n_labels = config["n_labels"]
        self.nearest_neighbors = config["max_neighbors"]
        # metrics
        self.top_k_metrics = config["top_k_metrics"]
        # optimizer and scheduler
        self.weight_decay = config["weight_decay"]
        self.init_lr = config["init_lr"]
        self.patience = config["lr"]["patience"]
        self.cooldown = config['lr']['cooldown']
        self.natoms = config['natoms']
        self.weights = {'protein': 1.0, 'ppi': 1.0, 'abag': 1.0}
        self.lr_scheduler = "plateau"
        if "scheduler" in config:
                self.lr_scheduler = config["scheduler"]
                if config["scheduler"] == 'cosine':
                        self.warmup = 50000
                        self.max_iters = 20000000


        # default
        checkpoint_models = True
        n_checkpoints = 1
        if 'checkpoint' in config:
                if config['checkpoint'] == 'all':
                        checkpoint_models = True
                        n_checkpoints = self.layers
                if config['checkpoint'] == 'none':
                        checkpoint_models = False
                        n_checkpoints = self.layers
        
        self.proj_input = nn.Linear(self.n_labels + self.natoms, self.n_hidden)
        self.net = EnTransformer(
            num_tokens=None,
            dim=self.n_hidden,
            depth=self.layers,  # depth
            dim_head=self.n_hidden // self.n_heads,  # dimension per head
            heads=self.n_heads,  # number of heads
            neighbors=self.
            nearest_neighbors,  # only do attention between coordinates N nearest neighbors - set to 0 to turn off
            talking_heads=
            True,  # use Shazeer's talking heads https://arxiv.org/abs/2003.02436
            checkpoint=
            checkpoint_models,
            n_checkpoints=n_checkpoints,
            use_cross_product=
            True,  # use cross product vectors (idea by @MattMcPartlon)
            rel_pos_emb=True,  # encode positions - ordered set
            num_atoms=self.natoms
        )

        self.dropout = nn.Dropout(self.dropout)
        self.proj = nn.Linear(self.n_hidden * self.natoms, self.n_labels)
        metrics = MetricCollection([
            Accuracy(ignore_index=self.n_labels,
                     top_k=self.top_k_metrics).double()
        ])
        self.train_metrics = metrics.clone(prefix='train_').double()
        self.valid_metrics = metrics.clone(prefix='val_').double()
        self.train_accuracy_protein = LocalAccuracy(ignore_index=self.n_labels)
        self.train_accuracy_ppi = LocalAccuracy(ignore_index=self.n_labels)
        self.train_accuracy_abag_ppi = LocalAccuracy(ignore_index=self.n_labels)
        self.train_accuracy_abag = LocalAccuracy(ignore_index=self.n_labels)
        self.train_accuracy_ab = LocalAccuracy(ignore_index=self.n_labels)

        self.val_accuracy_protein = LocalAccuracy(ignore_index=self.n_labels)
        self.val_accuracy_ppi = LocalAccuracy(ignore_index=self.n_labels)
        self.val_accuracy_abag_ppi = LocalAccuracy(ignore_index=self.n_labels)
        self.val_accuracy_abag = LocalAccuracy(ignore_index=self.n_labels)
        self.val_accuracy_ab = LocalAccuracy(ignore_index=self.n_labels)
        self.test_accuracy = LocalAccuracy(ignore_index=self.n_labels)
        self.train_accuracy_base = Accuracy(ignore_index=self.n_labels,
                                            top_k=self.top_k_metrics,
                                            threshold=0.01)

    def forward(self, nfeats, coords, mask=None, pos_indices=None, return_attn=False):
        # not giving adj mat: using nearest neighbors and attention to determine neighbor weighting
        feats = self.proj_input(nfeats.double())
        if return_attn:
                rep, x, attn_dict = self.net(feats, coords, mask=mask,
                                             pos_indices=pos_indices,
                                             return_attn=return_attn)
        else:
                rep, x = self.net(feats, coords, mask=mask, pos_indices=pos_indices)
        num_res = rep.shape[1] // self.natoms
        feats = rep.reshape(rep.shape[0], num_res, self.natoms*rep.shape[2])
        feats = self.dropout(feats)
        feats = self.proj(feats)
        if return_attn:
                return feats.view(-1, self.n_labels), x, \
                rep.reshape(rep.shape[0], num_res, self.natoms*rep.shape[2]),\
                attn_dict
        else:
                return feats.view(-1, self.n_labels), x, \
                rep.reshape(rep.shape[0], num_res, self.natoms*rep.shape[2])

    def configure_optimizers(self):
        
        if self.lr_scheduler == "cosine":
                optimizer_adam = \
                torch.optim.Adam(self.parameters(),
                             lr=self.init_lr, weight_decay=self.weight_decay)
                lr_scheduler = {
                'scheduler': \
                CosineWarmupScheduler(optimizer=optimizer_adam, warmup=self.warmup, 
                                        max_iters=self.max_iters),
                'interval': 'step',
                'name': 'lr'
                }
        #elif self.lr_scheduler == "onecycle":
        #        optimizer_adam = \
        #        torch.optim.SGD(self.parameters(),
        #                     lr=self.init_lr, weight_decay=self.weight_decay)
        else:
                optimizer_adam = \
                torch.optim.Adam(self.parameters(),
                             lr=self.init_lr, weight_decay=self.weight_decay)
                lr_scheduler = {
                'scheduler': \
                torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam,
                                                                verbose=True,
                                                                min_lr=0.0000001,
                                                                cooldown=self.cooldown,
                                                                patience=self.patience,
                                                                factor=0.1),
                'monitor': 'val_loss', #apply on validation loss
                'name': 'lr'
                }
        return {'optimizer': optimizer_adam, 'lr_scheduler': lr_scheduler}

    def training_step(self, batch, batch_idx):

        total_loss = 0.0
        
        if 'protein' in batch:
            id, (nfeats, coords, edges, mask, labels, pos_indices) = batch['protein']
            y_hat, coords, _ = self(nfeats, coords, mask=mask, pos_indices=pos_indices)
            loss = torch.nn.functional.cross_entropy(y_hat,
                                                    labels,
                                                    ignore_index=self.n_labels,
                                                    reduction='mean')
            dim_ignore_index = torch.full((y_hat.shape[0], 1),
                                        float('-inf')).type_as(y_hat)
            y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)

            accuracy = self.train_accuracy_protein(y_hat_ignore_index, labels)
            self.log("train_loss_{}".format('protein'),
                    loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True)

            self.log('train_accuracy_{}'.format('protein'),
                    accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True)
            
            total_loss += loss
            
        if 'ppi' in batch:
            id, (nfeats, coords, edges, mask, labels, pos_indices) = batch['ppi']
            y_hat, coords, _ = self(nfeats, coords, mask=mask, pos_indices=pos_indices)
            loss = torch.nn.functional.cross_entropy(y_hat,
                                                    labels,
                                                    ignore_index=self.n_labels,
                                                    reduction='mean')
            dim_ignore_index = torch.full((y_hat.shape[0], 1),
                                        float('-inf')).type_as(y_hat)
            y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)

            accuracy = self.train_accuracy_ppi(y_hat_ignore_index, labels)
            
            self.log("train_loss_{}".format('ppi'),
                    loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True)

            self.log('train_accuracy_{}'.format('ppi'),
                    accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True)

            total_loss += loss

        if 'abag_ppi' in batch:
            id, (nfeats, coords, edges, mask, labels, pos_indices) = batch['abag_ppi']
            y_hat, coords, _ = self(nfeats, coords, mask=mask, pos_indices=pos_indices)
            loss = torch.nn.functional.cross_entropy(y_hat,
                                                    labels,
                                                    ignore_index=self.n_labels,
                                                    reduction='mean')
            dim_ignore_index = torch.full((y_hat.shape[0], 1),
                                        float('-inf')).type_as(y_hat)
            y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)

            accuracy = self.train_accuracy_abag_ppi(y_hat_ignore_index, labels)
            
            self.log("train_loss_{}".format('abag_ppi'),
                    loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True)

            self.log('train_accuracy_{}'.format('abag_ppi'),
                    accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True)

            total_loss += loss

        if 'abag' in batch:
            id, (nfeats, coords, edges, mask, labels, pos_indices) = batch['abag']
            y_hat, coords, _ = self(nfeats, coords, mask=mask, pos_indices=pos_indices)
            loss = torch.nn.functional.cross_entropy(y_hat,
                                                    labels,
                                                    ignore_index=self.n_labels,
                                                    reduction='mean')
            dim_ignore_index = torch.full((y_hat.shape[0], 1),
                                        float('-inf')).type_as(y_hat)
            y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)

            accuracy = self.train_accuracy_abag(y_hat_ignore_index, labels)
            
            self.log("train_loss_{}".format('abag'),
                    loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True)

            self.log('train_accuracy_{}'.format('abag'),
                    accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True)

            total_loss += loss

        for key in ['ab', 'afab', 'afsc']:
            if key in batch:
                id, (nfeats, coords, edges, mask, labels, pos_indices) = batch[key]
                y_hat, coords, _ = self(nfeats, coords, mask=mask, pos_indices=pos_indices)
                loss = torch.nn.functional.cross_entropy(y_hat,
                                                        labels,
                                                        ignore_index=self.n_labels,
                                                        reduction='mean')
                dim_ignore_index = torch.full((y_hat.shape[0], 1),
                                                float('-inf')).type_as(y_hat)
                y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)

                accuracy = self.train_accuracy_ab(y_hat_ignore_index, labels)
                
                self.log("train_loss_{}".format('ab'),
                        loss,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                        sync_dist=True)

                self.log('train_accuracy_{}'.format('ab'),
                        accuracy,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True)

                total_loss += loss

        self.log("train_loss",
                    total_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss = 0.0
        if 'protein' in batch:
            id, (nfeats, coords, edges, mask, labels, pos_indices) = batch['protein']
            y_hat, coords, _ = self(nfeats, coords, mask=mask, pos_indices=pos_indices)
            #print('out:', y_hat.shape, coords.shape)
            loss = torch.nn.functional.cross_entropy(y_hat,
                                                    labels,
                                                    ignore_index=self.n_labels,
                                                    reduction='mean')
            dim_ignore_index = torch.full((y_hat.shape[0], 1),
                                        float('-inf')).type_as(y_hat)
            y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)

            accuracy = self.val_accuracy_protein(y_hat_ignore_index, labels)
            self.log("val_loss_{}".format('protein'),
                    loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True)

            self.log('val_accuracy_{}'.format('protein'),
                    accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True)

            total_loss += loss
        
        if 'ppi' in batch:
            id, (nfeats, coords, edges, mask, labels, pos_indices) = batch['ppi']
            y_hat, coords, _ = self(nfeats, coords, mask=mask, pos_indices=pos_indices)
            loss = torch.nn.functional.cross_entropy(y_hat,
                                                    labels,
                                                    ignore_index=self.n_labels,
                                                    reduction='mean')
            dim_ignore_index = torch.full((y_hat.shape[0], 1),
                                        float('-inf')).type_as(y_hat)
            y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)

            accuracy = self.val_accuracy_ppi(y_hat_ignore_index, labels)
            self.log("val_loss_{}".format('ppi'),
                    loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True)

            self.log('val_accuracy_{}'.format('ppi'),
                    accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True)

            total_loss += loss
        
        if 'abag_ppi' in batch:
            id, (nfeats, coords, edges, mask, labels, pos_indices) = batch['abag_ppi']
            y_hat, coords, _ = self(nfeats, coords, mask=mask, pos_indices=pos_indices)
            loss = torch.nn.functional.cross_entropy(y_hat,
                                                    labels,
                                                    ignore_index=self.n_labels,
                                                    reduction='mean')
            dim_ignore_index = torch.full((y_hat.shape[0], 1),
                                        float('-inf')).type_as(y_hat)
            y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)

            accuracy = self.val_accuracy_abag_ppi(y_hat_ignore_index, labels)
            self.log("val_loss_{}".format('abag_ppi'),
                    loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True)

            self.log('val_accuracy_{}'.format('abag_ppi'),
                    accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True)

            total_loss += loss

        if 'abag' in batch:
            id, (nfeats, coords, edges, mask, labels, pos_indices) = batch['abag']
            y_hat, coords, _ = self(nfeats, coords, mask=mask, pos_indices=pos_indices)
            loss = torch.nn.functional.cross_entropy(y_hat,
                                                    labels,
                                                    ignore_index=self.n_labels,
                                                    reduction='mean')
            dim_ignore_index = torch.full((y_hat.shape[0], 1),
                                        float('-inf')).type_as(y_hat)
            y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)

            accuracy = self.val_accuracy_abag(y_hat_ignore_index, labels)
            self.log("val_loss_{}".format('abag'),
                    loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True)

            self.log('val_accuracy_{}'.format('abag'),
                    accuracy,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True)

            total_loss += loss
        
        for key in ['ab', 'afab', 'afsc']:
            if key in batch:
                id, (nfeats, coords, edges, mask, labels, pos_indices) = batch[key]
                
                y_hat, coords, _ = self(nfeats, coords, mask=mask, pos_indices=pos_indices)
                loss = torch.nn.functional.cross_entropy(y_hat,
                                                        labels,
                                                        ignore_index=self.n_labels,
                                                        reduction='mean')
                dim_ignore_index = torch.full((y_hat.shape[0], 1),
                                                float('-inf')).type_as(y_hat)
                y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)

                accuracy = self.val_accuracy_ab(y_hat_ignore_index, labels)
                self.log("val_loss_{}".format('ab'),
                        loss,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                        sync_dist=True)

                self.log('val_accuracy_{}'.format('ab'),
                        accuracy,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True)

                total_loss += loss
                
        self.log("val_loss",
                    total_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True)

        return total_loss


    def test_step(self, batch, batch_idx):

        id, (nfeats, coords, edges, mask, labels, pos_indices) = batch
        #print(id)
        #split_coords = coords.reshape(-1, self.natoms, coords.shape[1]//self.natoms, 3)
        #print(coords[0, :4,:])
        y_hat, coords, _ = self(nfeats, coords, mask=mask, pos_indices=pos_indices)
        loss = torch.nn.functional.cross_entropy(y_hat,
                                                labels,
                                                ignore_index=self.n_labels,
                                                reduction='mean')
        dim_ignore_index = torch.full((y_hat.shape[0], 1),
                                    float('-inf')).type_as(y_hat)
        y_hat_ignore_index = torch.cat([y_hat, dim_ignore_index], dim=-1)

        accuracy = self.test_accuracy(y_hat_ignore_index, labels)
        self.log("test_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True)

        self.log('test_accuracy',
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True)
    
        return loss
