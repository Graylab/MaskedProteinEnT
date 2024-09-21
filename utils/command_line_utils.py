import argparse
import os
from datetime import datetime


def _get_args():
    """Gets command line arguments"""
    desc = ('''
        ''')
    parser = argparse.ArgumentParser(
        description=desc)
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Number of proteins per batch')
    parser.add_argument('--seed', type=int, default=0, help='Manual seed')
    parser.add_argument('--h5_file', type=str, default='')
    parser.add_argument('--h5_file_protein', type=str, default='')
    parser.add_argument('--h5_file_ppi', type=str, default='')
    parser.add_argument('--h5_file_abag_ppi', type=str, default='')
    parser.add_argument('--h5_file_ab', type=str, default='')
    parser.add_argument('--h5_file_afab', type=str, default='')
    parser.add_argument('--h5_file_afsc', type=str, default='')
    
    parser.add_argument('--model', type=str, default='')

    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--masking_rate_max',
                        type=float,
                        default=0.4,
                        help='Max mask rate for one partner')
    parser.add_argument('--masking_rate_min',
                        type=float,
                        default=0.1,
                        help='Min mask rate for one partner')
    parser.add_argument('--mask_ab_region',
                        type=str,
                        default='',
                        help='mask ab region; h1,h2,h3,l1,l2,l3,cdrs')
    parser.add_argument('--contact_residues_only',
                        action='store_true',
                        default=False,
                        help='select only contact residues with --mask_ab_region')
    parser.add_argument('--antibody',
                        action='store_true',
                        default=False,
                        help='use with PPISampler')
    parser.add_argument('--num_gpus',
                        type=int,
                        default=1,
                        help='number of gpus')
    parser.add_argument('--num_procs',
                        type=int,
                        default=6,
                        help='number of cpu procs available per node')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=350,
                        help="max seq length from SCN Dataset and EGNN")
    parser.add_argument('--crop_sequences',
                        action='store_true',
                        default=False,
                        help="crop sequence/structure to --max_seq_len")
    parser.add_argument('--scn_casp_version',
                        type=int,
                        default=12,
                        help="sidechainnet casp version")
    parser.add_argument('--scn_sequence_similarity',
                        type=int,
                        default=30,
                        help="sequence similarity culling of dataset - 30, 50, 90")
    parser.add_argument('--protein_gmodel',
                        type=str,
                        default='egnn-trans-ma',
                        help='graph-model to use for protein masked model: eggn, egnn-trans')
    parser.add_argument('--from_pdb',
                        type=str,
                        default='',
                        help='load info from pdb file instead of h5 file')
    parser.add_argument('--pdb_file',
                        type=str,
                        default='',
                        help='pdb file with structure')
    parser.add_argument('--sequences_file',
                        type=str,
                        default='',
                        help='txt file with 1 sequence per line for scoring')
    parser.add_argument('--partners_json',
                        type=str,
                        default='',
                        help='dictionary mapping pdbfile basename to partner chains separated by underscore')
    parser.add_argument('--sample_temperatures',
                        type=str,
                        default='0.1,0.5',
                        help='comma separted string of temperatures to sample sequences')
    parser.add_argument('--num_samples',
                        type=str,
                        default='20',
                        help='string or comma separted string of number of samples')
    parser.add_argument('--mask_ab_indices',
                        type=str,
                        default='',
                        help='0-numbered Ab residue indices to design')
    parser.add_argument('--ids',
                        type=str,
                        default='',
                        help='comma separated subset of pdb ids to select from h5 dataset or pdb files')
    parser.add_argument('--file_with_selected_scn_ids_for_training',
                        type=str,
                        default='',
                        help='file to train on only a subset of ids from SCN dataset')
    parser.add_argument('--file_with_selected_scn_ids_for_testing',
                        type=str,
                        default='',
                        help='file to train on only a subset of ids from SCN dataset')
    parser.add_argument('--use_scn',
                        action='store_true',
                        default=False,
                        help='Use SidechainNet Dataset also')
    parser.add_argument('--atom_types',
                        type=str,
                        default='backbone',
                        help='Use coords for these atoms. options: backbone, all, backbone_and_cb, cb, ca')
    parser.add_argument('--partner_name',
                        type=str,
                        default='p0',
                        help='Set which partner in complex to design; options: p0, p1, Ab, Ag, p0p1, AbAg')

    #TRAINING ARGS
    #device
    parser.add_argument('--use_dp',
                        action='store_true',
                        default=False,
                        help='use multiple gpus with DataParallel')
    parser.add_argument(
        '--use_ddp',
        action='store_true',
        default=False,
        help='use DistributedDataParallel. Not implemented yet.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help=('The chance of entire channels being zeroed out '
                              'during training'))

    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='Number of epochs')
    parser.add_argument('--internal_checkpoints',
                        type=str,
                        default='default',
                        help='set entransformer internal checkpoint segments; default, all, none')
    parser.add_argument('--save_every',
                        type=int,
                        default=5,
                        help='Save model every X number of epochs.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Number of proteins per batch')
    parser.add_argument('--seed', type=int, default=0, help='Manual seed')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate for Adam')
    parser.add_argument('--lr_patience',
                        type=float,
                        default=10,
                        help='use with --modify_lr_on_plateau; number of epochs with no change')
    parser.add_argument('--lr_cooldown',
                        type=float,
                        default=4,
                        help='use with --modify_lr_on_plateau; cooldown period after reduce')
    parser.add_argument('--scheduler',
                        type=str,
                        default="plateau",
                        help='cosine or plateau')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.001,
                        help='weight decay for Adam/Noam optimizer')
    
    parser.add_argument('--train_split',
                        type=float,
                        default=0.90,
                        help=('The percentage of the dataset that is used '
                              'during training'))
    
    parser.add_argument('--model', type=str, default='')

    parser.add_argument('--heads',
                        type=int,
                        default=4,
                        help='number of heads in GMHA')
    parser.add_argument('--layers',
                        type=int,
                        default=4,
                        help='number of layers in GMHA')
    parser.add_argument(
        '--model_dim',
        type=int,
        default=16,
        help='total model dimensions (per-head*heads)=model_dim')
    parser.add_argument(
        '--model_dim_edges',
        type=int,
        default=32,
        help='total model dimensions (per-head*heads)=model_dim')
    parser.add_argument('--max_ab_neighbors',
                        type=int,
                        default=15,
                        help='max number of nearest neighbors for antigen')
    parser.add_argument('--max_ag_neighbors',
                        type=int,
                        default=10,
                        help='max number of nearest neighbors for antigen')
    parser.add_argument('--full_connectivity',
                        action='store_true',
                        default=False,
                        help='use fully connected graph')

    #Output options
    parser.add_argument('--max_train',
                        type=int,
                        default=None,
                        help='pick max number of data point for dataset')
    parser.add_argument('--max_val',
                        type=int,
                        default=None,
                        help='pick max number of data point for dataset')
    parser.add_argument('--use_train_dataset',
                        action='store_true',
                        default=False,
                        help='use train dataset loader for prediction')
    now = str(datetime.now().strftime('%y%m%d_%H%M%S'))
    default_model_path = os.path.join(project_path,
                                      'trained_models/model_{}/'.format(now))
    parser.add_argument('--output_dir', type=str, default=default_model_path)
    parser.add_argument('--masking_rate_max',
                        type=float,
                        default=0.4,
                        help='Max mask rate for one partner')
    parser.add_argument('--masking_rate_min',
                        type=float,
                        default=0.1,
                        help='Min mask rate for one partner')
    parser.add_argument('--mask_ab_region',
                        type=str,
                        default='',
                        help='mask ab region; h1,h2,h3,l1,l2,l3,cdrs')å
    parser.add_argument('--num_gpus',
                        type=int,
                        default=1,
                        help='number of gpus')
    parser.add_argument('--num_procs',
                        type=int,
                        default=6,
                        help='number of cpu procs available per node')
    parser.add_argument('--topk_metrics',
                        type=int,
                        default=1,
                        help="get best of top N metrics")
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=350,
                        help="max seq length from SCN Dataset and EGNN")
    parser.add_argument('--concat_ab_datasets',
                        action='store_true',
                        default=False,
                        help="concat different antibody-only datasets to fit in gpu")
    parser.add_argument('--crop_sequences',
                        action='store_true',
                        default=False,
                        help="crop sequence/structure to --max_seq_len")
    parser.add_argument('--scn_casp_version',
                        type=int,
                        default=12,
                        help="sidechainnet casp version")
    parser.add_argument('--scn_sequence_similarity',
                        type=int,
                        default=30,
                        help="sequence similarity culling of dataset - 30, 50, 90")
    parser.add_argument('--protein_gmodel',
                        type=str,
                        default='egnn',
                        help='graph-model to use for protein masked model: eggn, egnn-trans')
    parser.add_argument('--from_pdb',
                        type=str,
                        default='',
                        help='load info from pdb file instead of h5 file')
    parser.add_argument('--ppi_partners_json',
                        type=str,
                        default='',
                        help='dictionary mapping pdbfile basename to partner chains separated by underscore')
    parser.add_argument('--sample_temperatures',
                        type=str,
                        default='0.1,0.2,1.0',
                        help='comma separted string of temperatures to sample sequences')
    parser.add_argument('--num_samples',
                        type=str,
                        default='100',
                        help='string or comma separted string of number of samples')
    parser.add_argument('--mask_ab_indices',
                        type=str,
                        default='',
                        help='0-numbered Ab residue indices to design')
    parser.add_argument('--ids',
                        type=str,
                        default='',
                        help='comma separated subset of pdb ids to select from h5 dataset or pdb files')
    parser.add_argument('--predict_from_esm',
                        action='store_true',
                        default=False,
                        help='predict masked label from esm')
    parser.add_argument('--file_with_selected_scn_ids_for_training',
                        type=str,
                        default='',
                        help='file to train on only a subset of ids from SCN dataset')
    parser.add_argument('--file_with_selected_scn_ids_for_testing',
                        type=str,
                        default='',
                        help='file to train on only a subset of ids from SCN dataset')
    parser.add_argument('--fine_tune',
                        action='store_true',
                        default=False,
                        help='continue from pretrained model fr fine-tuning')
    parser.add_argument('--use_scn',
                        action='store_true',
                        default=False,
                        help='Use SidechainNet Dataset also')
    parser.add_argument('--use_scn_valid_and_test',
                        action='store_true',
                        default=False,
                        help='Use SidechainNet Valid and Test sets')
    parser.add_argument('--inference',
                        action='store_true',
                        default=False,
                        help='Run in inference mode')
    parser.add_argument('--predict_coords',
                        action='store_true',
                        default=False,
                        help='predict coords')
    parser.add_argument('--use_pseudo_cb',
                        action='store_true',
                        default=False,
                        help='use pseudo cbeta')
    parser.add_argument('--disable_predict_seq',
                        action='store_true',
                        default=False,
                        help='disable seq pred. Use with predict_coords; default is to predict seq')
    parser.add_argument('--add_noise',
                        type=float,
                        default=0.0,
                        help='add gaussian noise to coord inputs')
    parser.add_argument('--clip_gradients',
                        action='store_true',
                        default=False,
                        help='clip gradients by norm')
    parser.add_argument('--seq_struct_loss_weights',
                        type=str,
                        default='',
                        help='specify seq-structure loss weights: e.g. 0.5,0.5')
    parser.add_argument('--masked_protein_base_model',
                        type=str,
                        default='',
                        help='ckpt file for masked protein model for fine-tuned ppi model')
    parser.add_argument('--adj_sparse',
                        action='store_true',
                        default=False,
                        help='Use SidechainNet Dataset also')
    parser.add_argument('--atom_types',
                        type=str,
                        default='backbone',
                        help='Use coords for these atoms. options: backbone, all, backbone_and_cb, cb, ca')
    parser.add_argument('--lightning_save_last_model',
                        action='store_true',
                        default=False,
                        help='option to ModelCheckpoint callback of pytorch lightning')
    return parser.parse_args()

