import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import matplotlib
import seaborn as sns
from src.data.constants import _aa_dict

def letter_to_num(string, dict_):
    """Function taken from ProteinNet (https://github.com/aqlaboratory/proteinnet/blob/master/code/text_parser.py).
    Convert string of letters to list of ints"""
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [int(i) for i in num_string.split()]
    return num

def num_to_letter(array, dict_=_aa_dict):
    dict_rev = {}
    for key in dict_:
        dict_rev[int(dict_[key])]=key
    seq_array = [dict_rev[t] for t in array]
    return ''.join(seq_array)


def sequences_to_probabilities(sequences: list, mode: str = 'prob'):
    '''
    sequences: N strings of length shape L
    '''
    seqs_num = np.array([letter_to_num(seq, _aa_dict) for seq in sequences])
    seqs_oh = torch.nn.functional.one_hot(torch.tensor(seqs_num).long(),
                        num_classes=20).permute(1, 2, 0)
    print(seqs_oh.shape)


    if mode == 'prob':
        seq_oh_avg = seqs_oh.sum(dim=-1) / float(seqs_oh.shape[-1])

        return np.array(seq_oh_avg)
    else:
        raise KeyError('mode {} not available'.format(mode))


def plot_seq_logo(seqs_prob, indices, outfile, negative=False, wt_seq=None,
                 highlight_positions=[]):
    import logomaker

    columns_logo_aa = list(_aa_dict.keys())
    df_logo = pd.DataFrame(data=seqs_prob,
                           index=range(1, seqs_prob.shape[0] + 1),
                           columns=columns_logo_aa)
    #print(df_logo)
    fig = plt.figure(figsize=((len(indices))*0.6,1.5*3))
    ax = plt.gca()
    if negative:
        ss_logo = logomaker.Logo(df_logo,
                             font_name='Stencil Std',
                             color_scheme='NajafabadiEtAl2017',
                             vpad=.1,
                             width=.8,
                             ax=ax,
                             shade_below=.5,
                             fade_below=.5)
    else:
        ss_logo = logomaker.Logo(df_logo,
                             font_name='Stencil Std',
                             color_scheme='NajafabadiEtAl2017',
                             vpad=.1,
                             width=.8,
                             ax=ax)
    #print(indices)
    if wt_seq != '':
        ss_logo.style_glyphs_in_sequence(sequence=wt_seq, color='darkgrey')
    if len(highlight_positions) > 0:
        for pos in highlight_positions:
            if pos in indices:
                ss_logo.highlight_position(p=indices.index(pos)+1, color='gold', alpha=.5)
    
    ss_logo.ax.set_xticks(range(1, len(indices) + 1))
    ss_logo.ax.set_xticklabels(indices, fontsize=20)
    plt.xticks(rotation=45)
    ss_logo.ax.yaxis.set_ticks_position('none')
    ss_logo.ax.yaxis.set_tick_params(pad=-1)
        
    plt.box(False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=600, transparent=True)
    plt.close()


