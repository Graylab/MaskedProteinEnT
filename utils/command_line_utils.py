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
    parser.add_argument('--partners_json',
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
    return parser.parse_args()

