from Bio import SeqIO
from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser
from src.data.constants import letter_to_num, _aa_dict, _aa_3_1_dict, num_to_letter
from src.data.utils.pdb import get_pdb_atoms
from os.path import basename, splitext
import logging
from Bio.PDB.Polypeptide import PPBuilder

def get_id(pdb_file_path):
    return splitext(basename(pdb_file_path))[0]


def get_chain_seqs(fasta_file_path, 
                   additional_chain_ids=[],
                   antigen_present=False):
    """Gets the sequnce of each chain in a fasta file
    :param fasta_file_path: The fasta file to read in.
    :return:
        A dictionary where the key is the chain id and the value is a list of
        ints corresponding to their amino acid.
    :rtype: dict
    """
    seqs = dict()
    
    for chain in SeqIO.parse(fasta_file_path, 'fasta'):
        if ':H' in chain.id or 'heavy' in chain.id:
            id_ = 'H'
        elif ':L' in chain.id or 'light' in chain.id:
            id_ = 'L'
        else:
            if antigen_present and (chain.id in additional_chain_ids):
                id_ = chain.id
            else:
                chain_id = str(chain.id).split(':')[1]
                msg = ('Expected a heavy chain or light chain, marked as \'H\' '
                   ' or \'L\'. Got a chain id of :{} from protein {}')
                raise ValueError(msg.format(chain_id, chain.id))

        seqs.update({id_: letter_to_num(str(chain.seq), _aa_dict)})
    return seqs


def get_cdr_indices(pdb_file_path):
    """Gets the indices of the CDR loop residues in the PDB file

    :param pdb_file_path: The pdb file to read.
    :return:
        A dictionary where the key is the loop name (h1, h2, h3, l1, l2, l3)
        and the value is a 2-tuple of the index range of residues in the loop.
    :rtype: dict
    """
    cdr_ranges = {
        'h1': [26, 35],
        'h2': [50, 65],
        'h4': [71, 78],
        'h3': [95, 102],
        'l1': [24, 34],
        'l2': [50, 56],
        'l4': [66, 71],
        'l3': [89, 97]
    }

    # Remove duplicate chainIDs (i.e. remove all ATOMS except for the first of
    # each chain) and reindex at 0
    data = get_pdb_atoms(pdb_file_path)
    data = data.drop_duplicates(
        ['chain_id', 'residue_num',
         'code_for_insertions_of_residues']).reset_index()

    # Get the 3 letter residue and residue ID for all the residues in the heavy chain
    heavy_chain_residues = data[data.chain_id == 'H']
    light_chain_residues = data[data.chain_id == 'L']

    # Extract the residues within the h3_cdr_range
    heavy_residue_nums = heavy_chain_residues.residue_num.astype('int32')
    h1_idxs = list(heavy_chain_residues[heavy_residue_nums.isin(
        cdr_ranges['h1'])].index)
    h2_idxs = list(heavy_chain_residues[heavy_residue_nums.isin(
        cdr_ranges['h2'])].index)
    h3_idxs = list(heavy_chain_residues[heavy_residue_nums.isin(
        cdr_ranges['h3'])].index)

    light_residue_nums = light_chain_residues.residue_num.astype('int32')
    l1_idxs = list(light_chain_residues[light_residue_nums.isin(
        cdr_ranges['l1'])].index)
    l2_idxs = list(light_chain_residues[light_residue_nums.isin(
        cdr_ranges['l2'])].index)
    l3_idxs = list(light_chain_residues[light_residue_nums.isin(
        cdr_ranges['l3'])].index)

    return dict(h1=h1_idxs,
                h2=h2_idxs,
                h3=h3_idxs,
                l1=l1_idxs,
                l2=l2_idxs,
                l3=l3_idxs)

def is_good_sequence(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure(get_id(pdb_file), pdb_file)
    seq=''
    for chain in structure.get_chains():
        seq += seq1(''.join([residue.resname for residue in chain]))
    #print(seq)
    if 'X' in seq:
        print("Bad sequence: ",seq)
        return False
    return True

def check_HL_chains(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure(get_id(pdb_file), pdb_file)
    ids=[]
    bH = False
    bL = False
    for chain in structure.get_chains():
        ids.append(chain.id)
    if ('H' in ids):
        bH = True
    if ('L' in ids):
        bL = True
    return bH,bL

def check_pdb_file(pdb_file,h_len,l_len, ag_chains_ids, ab_chains_dict=None):
    parser = PDBParser()
    structure = parser.get_structure(get_id(pdb_file), pdb_file)
    ids=[]
    bH = False
    bL = False
    for chain in structure.get_chains():
        ids.append(chain.id)
    h_chain_id = 'H'
    l_chain_id = 'L'
    if ab_chains_dict is not None:
        if 'H' in ab_chains_dict:
            h_chain_id = ab_chains_dict['H']
        if 'L' in ab_chains_dict:
            l_chain_id = ab_chains_dict['L']
    if (h_chain_id in ids):
        bH = True
    if (l_chain_id in ids):
        bL = True
    seq=''
    bgood=True
    blen=True
    ag_seq_len=0
    h_seq_len=0
    l_seq_len=0
    for chain in structure.get_chains():
        id_ = chain.id
        seq_cur = seq1(''.join([residue.resname for residue in chain]))
        seq += seq_cur
        #dont check for ag len - we only use ag fragments not full ag
        if id_ in ag_chains_ids:
            ag_seq_len += len(seq_cur)
        if id_ == h_chain_id:
            h_seq_len = len(seq_cur)
        if id_ == l_chain_id:
            l_seq_len = len(seq_cur)
    if h_seq_len>h_len or l_seq_len>l_len:
        blen=False
    if 'X' in seq:
        print("Bad sequence: ",seq)
        bgood=False
    
    return bH,bL,bgood,blen,h_seq_len,l_seq_len, ag_seq_len


def get_chain_seqs_from_pdb(pdb_file, antigen_info=False, ab_chains=['H', 'L'], 
                            skip_nonstandard=True):
    
    chain_seqs = dict()
    parser = PDBParser()
    ppb=PPBuilder()
    structure = parser.get_structure(get_id(pdb_file), pdb_file)
    
    for chain in structure.get_chains():
        id_ = chain.id
        seq = seq1(''.join([residue.resname for residue in chain\
                    if 'CA' in residue]))
        print(seq)
        if id_ not in ab_chains and (not antigen_info):
            continue
        if 'X' in seq:
            if skip_nonstandard:
                continue
            seqlist = [t for t in seq]
            x_ids = [i for i,aa in enumerate(seq) if aa=='X']
            for pp in ppb.build_peptides(chain, aa_only=False):
                start_id, stop_id = pp[0].get_id()[1], pp[-1].get_id()[1]
                for x_id in x_ids:
                    if (x_id <= stop_id) and (x_id >= start_id):
                        xid_rel = x_id - start_id
                        nonstd_aa = pp[xid_rel].get_resname()
                        if nonstd_aa in _aa_3_1_dict:
                            seqlist[x_id] = _aa_3_1_dict[nonstd_aa]
                        else:
                            # only for carb-prot dataset
                            logger = logging.getLogger(__name__ + '_non_standard_aa_')
                            logger.info('{}, {}, {}, {}\n'.format(get_id(pdb_file), id_, x_id, pdb_file))
                            seqlist[x_id] = 'A'
            seq = ''.join(seqlist)
            assert 'X' not in seq
        print(seq)
        chain_seqs.update({id_: letter_to_num(seq, _aa_dict)})
    return chain_seqs
