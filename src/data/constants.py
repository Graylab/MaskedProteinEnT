import re

_aa_dict = {
    'A': '0',
    'C': '1',
    'D': '2',
    'E': '3',
    'F': '4',
    'G': '5',
    'H': '6',
    'I': '7',
    'K': '8',
    'L': '9',
    'M': '10',
    'N': '11',
    'P': '12',
    'Q': '13',
    'R': '14',
    'S': '15',
    'T': '16',
    'V': '17',
    'W': '18',
    'Y': '19'
}

_aa_1_3_dict = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR',
    '-': 'GAP'
}

_aa_3_1_dict = {_aa_1_3_dict[key]:key for key in _aa_1_3_dict}
# From Rosetta Canonical amino acid modifications (individual patch cases)
_aa_3_1_dict['HYP'] = 'H'
_aa_3_1_dict['SCY'] = 'C'
_aa_3_1_dict['SEP'] = 'S'
_aa_3_1_dict['TPO'] = 'T'
_aa_3_1_dict['TYI'] = 'Y'
_aa_3_1_dict['TYS'] = 'Y'
_aa_3_1_dict['MLY'] = 'K'
_aa_3_1_dict['MLZ'] = 'K'
_aa_3_1_dict['FME'] = 'M'
_aa_3_1_dict['PTR'] = 'Y'
_aa_3_1_dict['M3L'] = 'K'
_aa_3_1_dict['HIC'] = 'H'


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
