import time
import sys
import io
import requests
import os
from os.path import splitext, basename
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1
from Bio import SeqIO
from bisect import bisect_left, bisect_right
import torch
import numpy as np
import pandas as pd
from src.data.utils.geometry import calc_dist_mat, get_masked_mat

def renumber_seq(chain_seq, scheme="-c"):
    success = False
    time.sleep(1)
    for i in range(10):
        try:
            response = requests.post(
                    'http://www.bioinf.org.uk/abs/abnum/abnum.cgi',
                    data={
                        "plain": "1",
                        "scheme": scheme,
                        "aaseq": chain_seq}
                    )
            success = response.status_code == 200 and not ("<html>"
                                                           in response.text)

            if success:
                break
            else:
                time.sleep((i + 1) * 5)
        except requests.exceptions.ConnectionError:
            time.sleep(60)

    # if success:
    numbering = response.text
    return numbering


def renumber_pdb(old_pdb, renum_pdb):
    success = False
    time.sleep(5)
    for i in range(10):
        try:
            with open(old_pdb, 'rb') as f:
                response = requests.post(
                    'http://www.bioinf.org.uk/abs/abnum/abnumpdb.cgi',
                    params={
                        "plain": "1",
                        "output": "-HL",
                        "scheme": "-c"
                    },
                    files={"pdb": f})

            success = response.status_code == 200 and not ("<html>"
                                                           in response.text)

            if success:
                break
            else:
                time.sleep((i + 1) * 5)
        except requests.exceptions.ConnectionError:
            time.sleep(60)

    # if success:
    new_pdb_data = response.text
    with open(renum_pdb, "w") as f:
        f.write(new_pdb_data)


def get_pdb_chain_seq(pdb_file, chain_id):
    raw_fasta = pdb2fasta(pdb_file)
    fasta = SeqIO.parse(io.StringIO(raw_fasta), 'fasta')
    chain_sequences = {
        chain.id.split(':')[1]: str(chain.seq)
        for chain in fasta
    }
    if chain_id not in chain_sequences.keys():
        print(
            "No such chain in PDB file. Chain must have a chain ID of \"[PDB ID]:{}\""
            .format(chain_id))
        return None
    return chain_sequences[chain_id]


def heavy_chain_seq(pdb_file):
    return get_pdb_chain_seq(pdb_file, chain_id="H")


def make_square_mask(v_mask):
    sq_mask = v_mask.expand((len(v_mask), len(v_mask)))
    sq_mask = sq_mask & sq_mask.transpose(0, 1)
    return sq_mask

def get_atom_coord(residue, atom_type):
    if atom_type in residue:
        return residue[atom_type].get_coord()
    else:
        return [0, 0, 0]


def get_cb_or_ca_coord(residue):
    if 'CB' in residue:
        return residue['CB'].get_coord()
    elif 'CA' in residue:
        return residue['CA'].get_coord()
    else:
        return [0, 0, 0]


def get_continuous_ranges(residues):
    """ Returns ranges of residues which are continuously connected (peptide bond length 1.2-1.45 Å) """
    dists = []
    for res_i in range(len(residues) - 1):
        dists.append(
            np.linalg.norm(
                get_atom_coord(residues[res_i], "C") -
                get_atom_coord(residues[res_i + 1], "N")))

    ranges = []
    start_i = 0
    for d_i, d in enumerate(dists):
        if d > 1.45 or d < 1.2:
            ranges.append((start_i, d_i + 1))
            start_i = d_i + 1
        if d_i == len(dists) - 1:
            ranges.append((start_i, None))

    return ranges


def place_fourth_atom(a_coord: torch.Tensor, b_coord: torch.Tensor,
                      c_coord: torch.Tensor, length: torch.Tensor,
                      planar: torch.Tensor,
                      dihedral: torch.Tensor) -> torch.Tensor:
    """
    Given 3 coords + a length + a planar angle + a dihedral angle, compute a fourth coord
    """
    bc_vec = b_coord - c_coord
    bc_vec = bc_vec / bc_vec.norm(dim=-1, keepdim=True)

    n_vec = (b_coord - a_coord).expand(bc_vec.shape).cross(bc_vec)
    n_vec = n_vec / n_vec.norm(dim=-1, keepdim=True)

    m_vec = [bc_vec, n_vec.cross(bc_vec), n_vec]
    d_vec = [
        length * torch.cos(planar),
        length * torch.sin(planar) * torch.cos(dihedral),
        -length * torch.sin(planar) * torch.sin(dihedral)
    ]

    d_coord = c_coord + sum([m * d for m, d in zip(m_vec, d_vec)])

    return d_coord

def place_missing_cb_o(atom_coords):
    cb_coords = place_fourth_atom(atom_coords['C'], atom_coords['N'],
                                  atom_coords['CA'], torch.tensor(1.522),
                                  torch.tensor(1.927), torch.tensor(-2.143))
    o_coords = place_fourth_atom(
        torch.roll(atom_coords['N'], shifts=-1, dims=0), atom_coords['CA'],
        atom_coords['C'], torch.tensor(1.231), torch.tensor(2.108),
        torch.tensor(-3.142))

    bb_mask = get_atom_coords_mask(atom_coords['N']) & get_atom_coords_mask(
        atom_coords['CA']) & get_atom_coords_mask(atom_coords['C'])
    missing_cb = (get_atom_coords_mask(atom_coords['CB']) & bb_mask) == 0
    atom_coords['CB'][missing_cb] = cb_coords[missing_cb]

    bb_mask = get_atom_coords_mask(
        torch.roll(
            atom_coords['N'], shifts=-1, dims=0)) & get_atom_coords_mask(
                atom_coords['CA']) & get_atom_coords_mask(atom_coords['C'])
    missing_o = (get_atom_coords_mask(atom_coords['O']) & bb_mask) == 0
    atom_coords['O'][missing_o] = o_coords[missing_o]


def get_atom_coords(pdb_file, chains=None):
    p = PDBParser()
    file_name = splitext(basename(pdb_file))[0]
    structure = p.get_structure(file_name, pdb_file)
    structure = structure[0]
    
    if chains is None:
        residues = [r for r in structure.get_residues()  if 'CA' in r]
    else:
        residues = []
        for chain in chains:
            for chain_s in structure.get_chains():
                id_ = chain_s.id
                if id_ == chain: 
                    residues += [r for r in chain_s.get_residues() if 'CA' in r]

    n_coords = torch.tensor([get_atom_coord(r, 'N') for r in residues])
    ca_coords = torch.tensor([get_atom_coord(r, 'CA') for r in residues])
    c_coords = torch.tensor([get_atom_coord(r, 'C') for r in residues])
    cb_coords = torch.tensor([get_atom_coord(r, 'CB') for r in residues])
    cb_ca_coords = torch.tensor([get_cb_or_ca_coord(r) for r in residues])
    o_coords = torch.tensor([get_atom_coord(r, 'O') for r in residues])

    atom_coords = {}
    atom_coords['N'] = n_coords
    atom_coords['CA'] = ca_coords
    atom_coords['C'] = c_coords
    atom_coords['CB'] = cb_coords
    atom_coords['CBCA'] = cb_ca_coords
    atom_coords['O'] = o_coords

    try:
        place_missing_cb_o(atom_coords)
    except:
        print('could not place missing atoms')

    return atom_coords


def get_atom_coords_mask(coords):
    mask = torch.ByteTensor([1 if sum(_) != 0 else 0 for _ in coords])
    mask = mask & (1 - torch.any(torch.isnan(coords), dim=1).byte())
    return mask


def protein_dist_coords_matrix(pdb_file,
                              mask=None,
                              mask_fill_value=-999,
                              device=None,
                              chains=None):
    atom_coords = get_atom_coords(pdb_file, chains=chains)
    
    atom_coords = get_atom_coords(pdb_file)

    n_coords = atom_coords['N']
    ca_coords = atom_coords['CA']
    cb_coords = atom_coords['CB']
    c_coords = atom_coords['C']
    cb_ca_coords = atom_coords['CBCA']
    seq_len = len(ca_coords)
    if mask is None:
        mask = torch.ones(seq_len).byte()
    
    n_mask = get_atom_coords_mask(n_coords)
    ca_mask = get_atom_coords_mask(ca_coords)
    cb_mask = get_atom_coords_mask(cb_coords)
    c_mask = get_atom_coords_mask(c_coords)
    cb_ca_mask = get_atom_coords_mask(cb_ca_coords)
    
    cb_coords[(mask & cb_mask)==0] = mask_fill_value
    ca_coords[(mask & ca_mask)==0] = mask_fill_value
    n_coords[(mask & n_mask)==0] = mask_fill_value
    c_coords[(mask & c_mask)==0] = mask_fill_value
    cb_coords[(mask & cb_mask)==0] = mask_fill_value

    seq_len = len(ca_coords)
    if mask is None:
        mask = torch.ones(seq_len).byte()

    bk_and_cb_coords = torch.stack([n_coords, ca_coords, c_coords, cb_coords])
    dist_mat = get_masked_mat(calc_dist_mat(cb_ca_coords, cb_ca_coords),
                              mask=make_square_mask(mask & cb_ca_mask),
                              mask_fill_value=mask_fill_value,
                              device=device)
    return dist_mat, bk_and_cb_coords


def get_residue_numbering_for_pdb(pdb_file):
    
    p = PDBParser()
    file_name = splitext(basename(pdb_file))[0]
    structure = p.get_structure(file_name, pdb_file)
    structure = structure[0]
    residues = {}
    for chain_s in structure.get_chains():
        id_ = chain_s.id
        residues[id_] = [r.get_id()[1] for r in chain_s.get_residues()
                        if 'CA' in r]
    return residues


def cdr_indices_from_chothia_numbering(residue_id_nums, cdr, h_len, chain_id):
    """Gets the index of a given CDR loop from residue numbering"""
    cdr_chothia_range_dict = {
        "h1": (26, 32),
        "h2": (52, 56),
        "h4": (71,78),
        "h3": (95, 102),
        "l1": (24, 34),
        "l2": (50, 56),
        "l4": (66,71),
        "l3": (89, 97)
    }
    assert cdr in cdr_chothia_range_dict.keys()

    chothia_range = cdr_chothia_range_dict[cdr]
    

    # Binary search to find the start and end of the CDR loop
    cdr_start = bisect_left(residue_id_nums, chothia_range[0])
    cdr_end = bisect_right(residue_id_nums, chothia_range[1]) - 1

    if chain_id == "L":
        cdr_start += h_len
        cdr_end += h_len

    return cdr_start, cdr_end

def pdb2fasta(pdb_file, num_chains=None):
    """Converts a PDB file to a fasta formatted string using its ATOM data"""
    pdb_id = basename(pdb_file).split('.')[0]
    parser = PDBParser()
    structure = parser.get_structure(pdb_id, pdb_file)
    structure = structure[0]

    real_num_chains = len([0 for _ in structure.get_chains()])
    if num_chains is not None and num_chains != real_num_chains:
        print('WARNING: Skipping {}. Expected {} chains, got {}'.format(
            pdb_file, num_chains, real_num_chains))
        return ''

    fasta = ''
    for chain in structure.get_chains():
        id_ = chain.id
        seq = seq1(''.join([residue.resname for residue in chain]))
        fasta += '>{}:{}\t{}\n'.format(pdb_id, id_, len(seq))
        max_line_length = 80
        for i in range(0, len(seq), max_line_length):
            fasta += f'{seq[i:i + max_line_length]}\n'
    return fasta


def cdr_indices(chothia_pdb_file, cdr):
    """Gets the index of a given CDR loop"""
    
    cdr = str.lower(cdr)
    chain_id = cdr[0].upper()

    parser = PDBParser()
    pdb_id = os.path.basename(chothia_pdb_file).split('.')[0]
    structure = parser.get_structure(pdb_id, chothia_pdb_file)
    structure = structure[0]
    cdr_chain_structure = None
    for chain in structure.get_chains():
        if chain.id == chain_id:
            cdr_chain_structure = chain
            break
    if cdr_chain_structure is None:
        return None

    residue_id_nums = [res.get_id()[1] for res in cdr_chain_structure]
    if len(get_pdb_chain_seq(chothia_pdb_file,
                             chain_id=chain_id)) != len(residue_id_nums):
        print('ERROR in PDB file ' + chothia_pdb_file)
        print('residue id len', len(residue_id_nums))
        print('seq', len(heavy_chain_seq(chothia_pdb_file)))

    # Binary search to find the start and end of the CDR loop

    return cdr_indices_from_chothia_numbering(residue_id_nums, cdr,
                                             len(heavy_chain_seq(chothia_pdb_file)),
                                             chain_id)


def get_indices_dict_for_cdrs(target_pdb, cdr_loops="h1,h2,h3,l1,l2,l3", per_chain_dict=True):

    """Generates a list of indices for all residues that are part of the specified CDR loops.
    Args:
        target_pdb (string): path to pdb file
        cdr_loops (string): comma-seperated CDR loops, for which indices are needed.

    Returns:
        cdr_indices_list(list): List of residue indices (int) that are part of the specified CDR loops.
    """

    cdrs_to_design = cdr_loops.split(',')

    cdr_indices_dict = {}
    for cdr in cdrs_to_design:
        cdr_range = cdr_indices(target_pdb, cdr)
        if cdr_range is None:
            continue
        if per_chain_dict:
            key = cdr[0].upper()
        else:
            key = cdr
        if key in cdr_indices_dict:
            cdr_indices_dict[key] += [t for t in range(cdr_range[0], cdr_range[1] + 1)]
        else:
            cdr_indices_dict[key] = [t for t in range(cdr_range[0], cdr_range[1] + 1)]

    return cdr_indices_dict


def get_pdb_atoms(pdb_file_path):
    """Returns a list of the atom coordinates, and their properties in a pdb file
    :param pdb_file_path:
    :return:
    """
    with open(pdb_file_path, 'r') as f:
        lines = [line for line in f.readlines() if 'ATOM' in line]
    column_names = [
        'atom_num', 'atom_name', 'alternate_location_indicator',
        'residue_name', 'chain_id', 'residue_num',
        'code_for_insertions_of_residues', 'x', 'y', 'z', 'occupancy',
        'temperature_factor', 'segment_identifier', 'element_symbol'
    ]
    # Get the index at which each column starts/ends
    column_ends = np.array(
        [3, 10, 15, 16, 19, 21, 25, 26, 37, 45, 53, 59, 65, 75, 77])
    column_starts = column_ends[:-1] + 1
    column_ends = column_ends[1:]  # Ignore the first column (just says 'ATOM')

    rows = [[
        l[start:end + 1].replace(' ', '')
        for start, end in zip(column_starts, column_ends)
    ] for l in lines]
    return pd.DataFrame(rows, columns=column_names)
