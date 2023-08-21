from os import listdir
from os.path import join


from deeph3.preprocess.antigen_parser import add_antigen_info, add_protein_info,\
                                             ag_chains_ids_mod
from deeph3.preprocess.ppi_parser import add_ppi_info
#from deeph3.util.rosetta_interface_utils import get_pair_map, get_partners
from deeph3.preprocess.fragment_utils import *
from deeph3.preprocess.parsing_utils import get_cdr_indices, get_chain_seqs,\
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


def get_info(pdb_file,
             fasta_file=None,
             antigen_info=False,
             distance_cutoff=14,
             loops_only_interface=True,
             include_epitope_neighbors=False,
             distance_cutoff_epitope=12,
             write_trunc_pdbs=False,
             use_ppi_data=False,
             partners='',
             max_length=350,
             min_interface_dist=False,
             use_pairwise_geometry=False,
             average_ha_dist=True,
             get_cb_coords=False,
             get_bk_cb_coords=False,
             get_fullatom=False,
             add_per_res=False,
             ab_chains=['H', 'L'],
             do_not_truncate=False
             ):

    info = {}
    if fasta_file is not None:
        chain_seqs = get_chain_seqs(fasta_file)
    else:
        try:
            chain_seqs = get_chain_seqs_from_pdb(pdb_file,
                                                 antigen_info=antigen_info,
                                                 ab_chains=ab_chains)
        except:
            return None
    
    if not use_ppi_data:
        cdr_indices = get_indices_dict_for_cdrs(pdb_file, per_chain_dict=False)
        info.update(cdr_indices)
        chain_lengths = {}
        for key in chain_seqs:
            chain_lengths[key] = len(chain_seqs[key])
        info.update(chain_seqs)
        id_ = get_id(pdb_file)
        info.update(dict(id=id_))
        if not antigen_info:
            info = add_protein_info(info, pdb_file, chain_seqs,
                                    use_pairwise_geometry=use_pairwise_geometry,
                                    get_cb_coords=get_cb_coords,
                                    get_bk_and_cb_coords=get_bk_cb_coords,
                                    get_fullatom=get_fullatom,
                                    add_per_res=add_per_res)
        else:
            info.update(dict(antigen_info=antigen_info))
            info = add_antigen_info(info, pdb_file, chain_seqs, cdr_indices,
                                    distance_cutoff=distance_cutoff,
                                    loops_only_interface=loops_only_interface,
                                    include_epitope_neighbors=include_epitope_neighbors,
                                    distance_cutoff_epitope=distance_cutoff_epitope,
                                    write_trunc_pdbs=write_trunc_pdbs,
                                    min_interface_dist=min_interface_dist,
                                    average_ha_dist=average_ha_dist,
                                    use_pairwise_geometry=use_pairwise_geometry,
                                    get_cb_coords=get_cb_coords,
                                    get_bk_and_cb_coords=get_bk_cb_coords,
                                    get_fullatom=get_fullatom,
                                    add_per_res=add_per_res
                                    )
    else:
        info.update(dict(use_ppi_data=use_ppi_data))
        info = add_ppi_info(info, pdb_file, chain_seqs,
                            distance_cutoff=distance_cutoff,
                            include_epitope_neighbors=include_epitope_neighbors,
                            distance_cutoff_epitope=distance_cutoff_epitope,
                            write_trunc_pdbs=write_trunc_pdbs,
                            partners=partners,
                            max_length=max_length,
                            min_interface_dist=min_interface_dist,
                            average_ha_dist=average_ha_dist,
                            use_pairwise_geometry=use_pairwise_geometry,
                            get_cb_coords=get_cb_coords,
                            get_bk_and_cb_coords=get_bk_cb_coords,
                            get_fullatom=get_fullatom,
                            add_per_res=add_per_res,
                            do_not_truncate=do_not_truncate
                            )
    
    return info
