from os import listdir
from os.path import join


from src.data.preprocess.antigen_parser import add_antigen_info, add_protein_info,\
                                             ag_chains_ids_mod
from src.data.preprocess.ppi_parser import add_ppi_info
from src.data.preprocess.parsing_utils import get_cdr_indices, get_chain_seqs,\
                                            get_chain_seqs_from_pdb, get_id


def antibody_db_seq_info(fasta_dir,antigen_present=False):
    fasta_files = [
        join(fasta_dir, _) for _ in listdir(fasta_dir) if _[-5:] == 'fasta'
    ]

    num_seqs = len(fasta_files)
    min_heavy_seq_len = min_light_seq_len = min_total_seq_len = float('inf')
    max_heavy_seq_len = max_light_seq_len = max_total_seq_len = -float('inf')
    if antigen_present:
        min_ag_seq_len = float('inf')
        max_ag_seq_len = -float('inf')

    for fasta_file in fasta_files:
        chains = list(SeqIO.parse(fasta_file, 'fasta'))
        if len(chains) != 2:
            msg = 'Expected 2 chains in {}, got {}'.format(
                fasta_file, len(chains))
            raise ValueError(msg)

        h_len = l_len = 0
        if antigen_present:
            ag_len = 0
        for chain in chains:
            if ':H' in chain.id or 'heavy' in chain.id:
                h_len = len(chain.seq)
                max_heavy_seq_len = max(h_len, max_heavy_seq_len)
                min_heavy_seq_len = min(h_len, min_heavy_seq_len)
            elif ':L' in chain.id or 'light' in chain.id:
                l_len = len(chain.seq)
                max_light_seq_len = max(l_len, max_light_seq_len)
                min_light_seq_len = min(l_len, min_light_seq_len)
            elif antigen_present:
                if chain.id in ag_chains_ids_mod:
                    ag_len += len(chain.seq)
            else:
                try:
                    chain_id = str(chain.id).split(':')[1]
                    msg = (
                        'Expected a heavy chain or light chain, marked as \'H\' '
                        ' or \'L\'. Got a chain id of :{} from protein {}')
                    raise ValueError(msg.format(chain_id, chain.id))
                except Exception:
                    raise ValueError(
                        '{} does not have >name:chain format'.format(
                            fasta_file))

        total_len = h_len + l_len
        max_total_seq_len = max(total_len, max_total_seq_len)
        min_total_seq_len = min(total_len, min_total_seq_len)
        
        if antigen_present:
            max_ag_seq_len = max(ag_len, max_ag_seq_len)
            min_ag_seq_len = min(ag_len, min_ag_seq_len)
            total_len = h_len + l_len + ag_len
            max_total_seq_len = max(total_len, max_total_seq_len)
            min_total_seq_len = min(total_len, min_total_seq_len)
            

    seq_info_dict = dict(num_seqs=num_seqs,
                max_heavy_seq_len=max_heavy_seq_len,
                min_heavy_seq_len=min_heavy_seq_len,
                max_light_seq_len=max_light_seq_len,
                min_light_seq_len=min_light_seq_len,
                max_total_seq_len=max_total_seq_len,
                min_total_seq_len=min_total_seq_len)
    
    if antigen_present:
        seq_info_dict.update(dict(
            max_ag_seq_len=max_ag_seq_len,
            min_ag_seq_len=min_ag_seq_len))

    return seq_info_dict


