from tqdm import tqdm
import os
import pandas as pd
import torch
from utils.command_line_utils import _get_args
from utils.AntibodyMetricsReporter import AntibodyMetricsReporter
    
torch.set_grad_enabled(False)

def get_trials(mr):
    if mr==1.0:
        trials_masking = 1
        trials_sampling = 100
    elif mr >= 0.50 and mr < 1.0:
        trials_masking = 20
        trials_sampling = 100
    else:
        # low mr
        trials_masking = 50
        trials_sampling = 20
    return trials_masking, trials_sampling


def run_ab_or_ag_masking(masking_rates=[1.0]):
    row_list = []
    for sel in ['Ab', 'Ag']:
        print('Selection partner', sel)
        for mr in tqdm(masking_rates):
            trials_masking, trials_sampling = get_trials(mr)
            print(mr, sel)
            reporter = AntibodyMetricsReporter(partner_selection=sel,
                                                region_selection=None,
                                                intersect_with_contacts=False)
            (outdict, iddict), (outdict_sampled, iddict_sampled, outdict_max_sampled) = \
                reporter.run_trials_for_seqrec(mr=mr,
                                                trials_masking=trials_masking,
                                                trials_sampling=trials_sampling,
                                                subdir='MaskAborAgContact'
                                                )
            
            for key in outdict:
                df_row_dict = {}
                df_row_dict['AndIntersection'] = True #default with not selecting region
                df_row_dict['Mask'] = sel
                df_row_dict['Region'] = key
                df_row_dict['Accuracy'] = outdict[key]
                df_row_dict['Trials_Masking'] = trials_masking
                df_row_dict['Trials_Sampling'] = trials_sampling
                df_row_dict['MaskRate'] = mr
                #df_row_dict['PerTarget'] = iddict[key]
                df_row_dict['Accuracy_Sampled'] = outdict_sampled[key]
                df_row_dict['Accuracy_SampledMax'] = outdict_max_sampled[key]
                #df_row_dict['PerTarget_Sampled'] = iddict_sampled[key]
                row_list.append(df_row_dict)
    return row_list

def run_region_masking(masking_rates=[1.0],
                       regions = ['h1', 'h2', 'h3', 'l1', 'l2', 'l3', 'non-cdr'],
                       contact_options = [True, False],
                       subdirs = ['MaskCDRandContact', 'MaskCDR']):
    partner_sel = 'Ab'
    row_list = []
    
    for intersect_with_contacts, subdir in zip(contact_options, subdirs):
        for region_sel in regions:
            print('Selection partner', partner_sel)
            for mr in tqdm(masking_rates):
                trials_masking, trials_sampling = get_trials(mr)
                print(partner_sel, mr, region_sel, intersect_with_contacts)
                reporter = AntibodyMetricsReporter(partner_selection=partner_sel,
                                                    region_selection=region_sel,
                                                    intersect_with_contacts=intersect_with_contacts)
                (outdict, iddict), (outdict_sampled, iddict_sampled, outdict_max_sampled) = \
                    reporter.run_trials_for_seqrec(mr=mr,
                                                    trials_masking=trials_masking,
                                                    trials_sampling=trials_sampling,
                                                    subdir=subdir,
                                                    suffix='_{}'.format(region_sel)
                                                    )
                df_row_dict = {}
                df_row_dict['AndIntersection'] = intersect_with_contacts
                df_row_dict['Mask'] = region_sel
                df_row_dict['Region'] = region_sel
                df_row_dict['Accuracy'] = outdict[region_sel]
                df_row_dict['Trials_Masking'] = trials_masking
                df_row_dict['Trials_Sampling'] = trials_sampling
                df_row_dict['MaskRate'] = mr
                #df_row_dict['PerTarget'] = iddict[region_sel]
                df_row_dict['Accuracy_Sampled'] = outdict_sampled[region_sel]
                df_row_dict['Accuracy_SampledMax'] = outdict_max_sampled[region_sel]
                #df_row_dict['PerTarget_Sampled'] = iddict_sampled[region_sel]
                row_list.append(df_row_dict)
    return row_list


args = _get_args()
do_not_overwrite=False
csv_file = os.path.join(args.output_dir, 'ConsolidatedAccuracy_MaskedCdrs.csv')
if os.path.exists(csv_file) and do_not_overwrite:
    df_2 = pd.read_csv(csv_file)
    if 'Unnamed: 0' in df_2.columns:
        df_2.drop(columns=['Unnamed: 0'], inplace=True)
    row_list = df_2.to_dict(orient='list')
else:
    row_list = run_region_masking()
    df_2 = pd.DataFrame(row_list)
    df_2.to_csv(csv_file, index=False)

csv_file = os.path.join(args.output_dir, 'ConsolidatedAccuracy_MaskedAborAg.csv')
if os.path.exists(csv_file) and do_not_overwrite:
    print('Found existing: ', csv_file)
    df_1 = pd.read_csv(csv_file)
    if 'Unnamed: 0' in df_1.columns:
        df_1.drop(columns=['Unnamed: 0'], inplace=True)
    print(df_1)
    row_list = df_1.to_dict(orient='list')
    print(row_list.keys())
else:
    row_list = run_ab_or_ag_masking()
    df_1 = pd.DataFrame(row_list)
    df_1.to_csv(csv_file, index=False)

df = pd.concat([df_1, df_2]).reset_index()
df.to_csv(os.path.join(args.output_dir, 'ConsolidatedAccuracy_AbAgRegions_All.csv'))