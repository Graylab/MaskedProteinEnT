from utils.metrics import score_sequences
from utils.prepare_model_inputs_from_pdb import get_protein_info_from_pdb_file,\
get_antibody_info_from_pdb_file
from src.data.constants import letter_to_num, _aa_dict
import torch

torch.set_default_dtype(torch.float64)
torch.set_grad_enabled(False)
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

def sequences_to_labels(sequence_file):
    with open(sequence_file, 'r') as f:
        sequences = [t.rstrip() for t in f.readlines()]
    labels = []
    for seq in sequences:
        labels.append(torch.tensor(letter_to_num(seq, _aa_dict)).long())
    
    return labels, sequences

def score_antibody_sequences(pdb_file, sequence_file, model, outfile='scores.csv'):
    batch = get_antibody_info_from_pdb_file(pdb_file)
    sequence_labels, sequences = sequences_to_labels(sequence_file)
    score_dict = score_sequences(batch, sequence_labels, model)
    df = pd.DataFrame()
    df['Sequences'] = sequences
    df['Scores'] = score_dict['scores']
    df.to_csv(outfile, index=False)

if __name__ == '__main__':
    args = _get_args()
    model = ProteinMaskedLabelModel_EnT_MA.load_from_checkpoint(args.model).to(device)
    model.freeze()
    outfile=os.path.join(args.output_dir, 'sequence_scores.csv')
    score_antibody_sequences(args.from_pdb, args.sequences_file, model, outfile)




