import argparse
import sys
import torch
import os
import timm
import pandas as pd
from interpretation_utils import draw_roc_curve, draw_confusion_matrix, calculate_probability, draw_confusion_matrix_prob_distribution
from interpretation_utils import draw_roc_curve_all, draw_pr_curve_all, majority_rule, calculate_metrics, bootstrap_ci, mcnemar_test
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

def Parser_main():
    parser = argparse.ArgumentParser(description="Bone fracture classification")
    parser.add_argument("--rootdir" , help="dataset rootdir", type = str)
    parser.add_argument("--script_dir", help = 'Location of the PathFxDx scripts rootdir', type = str)
    parser.add_argument("--cuda", default = 'cuda:0', type = str, help = 'cuda device or cpu')
    parser.add_argument("--random_seed", default = 1234567, type = int)
    parser.add_argument("--dataset", default = 'Dual', type = str)
    parser.add_argument("--analysis_dir", type = str)
    parser.add_argument("--model_dir", help = "result save directory", type = str)
    parser.add_argument("--model_name", default = 'fracture_best_model.pt', help = 'model name in model dir', type = str)
    parser.add_argument('--clinical_data', help = 'clinical_data')
    parser.add_argument('--bbox_size', default = None, type = int)
    parser.add_argument('--target_metric', default = 'AUPRC', type = str, help = 'Target metric for select best model')
    parser.add_argument('--expert_data', default = None,  help = 'expert_annotation_data', type = str)
    return parser.parse_args()

Argument = Parser_main()
sys.path.append(Argument.script_dir)
from Dataloader import xray_dataset_class_val
from model import CustomViT

meta_df_all = pd.read_excel(Argument.clinical_data)
data_root_dir = Argument.rootdir
root_dir = Argument.analysis_dir
dataset = Argument.dataset
root_dir_mode = os.path.join(root_dir, dataset, 'All')
model_dir = os.path.join(root_dir_mode, Argument.model_dir)
target_metric = Argument.target_metric

if Argument.model_name == None:
    model_path_list = os.listdir(os.path.join(model_dir, 'model'))
    if target_metric == 'AUROC':
        metric_list = [float(model_path.split('_')[2]) for model_path in model_path_list]
    else:
        metric_list = [float(model_path.split('_')[5]) for model_path in model_path_list]
    target_model_idx = np.argmax(np.array(metric_list))
    model_path = model_path_list[target_model_idx]
else:
    model_path = Argument.model_name

model_absolute_path = os.path.join(model_dir, 'model', model_path)

save_dir = os.path.join(model_dir, 'analyze')
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
save_dir = os.path.join(save_dir, 'result')
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)

meta_df = meta_df_all.copy()
snuh_df = meta_df[meta_df['Hospital'] == 'SNUH']
snuh_train_df = snuh_df[snuh_df['Internal_test_set'] == 0]
snuh_test_df = snuh_df[snuh_df['Internal_test_set'] == 1]
snubh_df = meta_df[meta_df['Hospital'] == 'SNUBH']
ncc_df = meta_df[meta_df['Hospital'] == 'NCC']
vincent_df = meta_df[meta_df['Hospital'] == 'Vincent']
external_df = meta_df[meta_df['Hospital'] != 'SNUH']
external_df = external_df[external_df['Hospital'] != 'BRMH']

print("Load classification dataset")
snuh_train_data = xray_dataset_class_val(snuh_train_df, data_root_dir)
snuh_test_data = xray_dataset_class_val(snuh_test_df, data_root_dir)
snubh_data = xray_dataset_class_val(snubh_df, data_root_dir)
ncc_data = xray_dataset_class_val(ncc_df, data_root_dir)
vincent_data = xray_dataset_class_val(vincent_df, data_root_dir)
external_data = xray_dataset_class_val(external_df, data_root_dir)

print("Load classification model")
if dataset == 'Dual':
    model = CustomViT()
else:
    model = timm.create_model('vit_base_patch16_224_dino', pretrained=False, num_classes=2)

state_dict = torch.load(model_absolute_path)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
device = torch.device(Argument.cuda)
model.to(device)
print("Loading model complete")

snuh_train_result_df = calculate_probability(model, snuh_train_data, device, dataset, meta_df)
snuh_test_result_df = calculate_probability(model, snuh_test_data, device, dataset, meta_df)
snubh_result_df = calculate_probability(model, snubh_data, device, dataset, meta_df)
ncc_result_df = calculate_probability(model, ncc_data, device, dataset, meta_df)
vincent_result_df = calculate_probability(model, vincent_data, device, dataset, meta_df)
external_result_df = calculate_probability(model, external_data, device, dataset, meta_df)

snuh_train_dir = os.path.join(save_dir, 'SNUH_Train')
snuh_test_dir = os.path.join(save_dir, 'SNUH_Test')
snubh_dir = os.path.join(save_dir, 'SNUBH')
ncc_dir = os.path.join(save_dir, 'NCC')
vincent_dir = os.path.join(save_dir, 'SVH')
external_dir = os.path.join(save_dir, 'External')

if os.path.exists(snuh_train_dir) is False:
    os.mkdir(snuh_train_dir)
if os.path.exists(snuh_test_dir) is False:
    os.mkdir(snuh_test_dir)
if os.path.exists(snubh_dir) is False:
    os.mkdir(snubh_dir)
if os.path.exists(ncc_dir) is False:
    os.mkdir(ncc_dir)
if os.path.exists(vincent_dir) is False:
    os.mkdir(vincent_dir)
if os.path.exists(external_dir) is False:
    os.mkdir(external_dir)

snuh_train_result_df.to_csv(os.path.join(save_dir, 'SNUH_Train', 'SNUH_Train_result.csv'), index = False)
snuh_test_result_df.to_csv(os.path.join(save_dir, 'SNUH_Test', 'SNUH_Test_result.csv'), index = False)
snubh_result_df.to_csv(os.path.join(save_dir, 'SNUBH', 'SNUBH_result.csv'), index = False)
ncc_result_df.to_csv(os.path.join(save_dir, 'NCC', 'NCC_result.csv'), index = False)
vincent_result_df.to_csv(os.path.join(save_dir, 'SVH', 'SVH_result.csv'), index=False)
external_result_df.to_csv(os.path.join(save_dir, 'External', 'External_result.csv'), index = False)

colors = {'SNUH_Train':'#D35400',
          'SNUH_Test':'#1F4E79',
          'SNUBH':'#C1443C',
          'NCC': '#76B041',
          'SVH': '#6B7B8C',
          'External': '#684D8A',
          'Whole_test': '#3D9991'}

snuh_train_threshold = draw_roc_curve(save_dir, 'SNUH_Train', color = colors['SNUH_Train'])
snuh_test_threshold = draw_roc_curve(save_dir, 'SNUH_Test', color = colors['SNUH_Train'])
_ = draw_roc_curve(save_dir, 'SNUBH', color = colors['SNUBH'])
_ = draw_roc_curve(save_dir, 'NCC', color = colors['NCC'])
_ = draw_roc_curve(save_dir, 'SVH', color = colors['SVH'])
_ = draw_roc_curve(save_dir, 'External', color = colors['External'])

name_target_list = ['SNUH_Test', 'External', 'SNUBH', 'NCC', 'SVH']
draw_roc_curve_all(save_dir, name_target_list, colors)
draw_pr_curve_all(save_dir, name_target_list, colors)

print('PathFxDx Thershold: ' + str(snuh_train_threshold))
metrics_dict = {}
snuh_test_metrics = draw_confusion_matrix(save_dir, 'SNUH_Test', snuh_train_threshold, colors['SNUH_Test'])
snubh_metrics = draw_confusion_matrix(save_dir, 'SNUBH', snuh_train_threshold, colors['SNUBH'])
ncc_metrics = draw_confusion_matrix(save_dir, 'NCC', snuh_train_threshold, colors['NCC'])
vincent_metrics = draw_confusion_matrix(save_dir, 'SVH', snuh_train_threshold, colors['SVH'])
external_metrics = draw_confusion_matrix(save_dir, 'External', snuh_train_threshold, colors['External'])
draw_confusion_matrix_prob_distribution(save_dir, ['External'], snuh_train_threshold, colors['Whole_test'])

metrics_dict['SNUH_test'] = snuh_test_metrics
metrics_dict['SNUBH'] = snubh_metrics
metrics_dict['NCC'] = ncc_metrics
metrics_dict['SVH'] = vincent_metrics
metrics_dict['Extneral_metrics'] = external_metrics

# Convert to DataFrame
df = pd.DataFrame(metrics_dict).transpose()  # Transpose to get rows and columns in expected orientation
df.to_csv(os.path.join(save_dir, 'final_metric.csv'), index_label = 'Hospital')

snuh_test_save_dir = os.path.join(save_dir, 'SNUH_Test')
sample_df = pd.read_csv(os.path.join(snuh_test_save_dir, 'SNUH_Test_result.csv'))
y_true = sample_df['Class'].to_list()
y_scores = sample_df['Probability'].to_list()
# Convert lists to numpy arrays for consistency
y_true = np.array(y_true)
y_scores = np.array(y_scores)
# Apply threshold to generate binary predictions
y_pred = (y_scores > snuh_train_threshold).astype(int)
sample_df['Model'] = y_pred
sample_df = sample_df.loc[:,['Image_Name', 'Model']]


if Argument.expert_data != None:
    expert_df = pd.read_csv(Argument.expert_data)
    expert_df.columns = ['Image_Name_x', 'Probability', 'Class', 'Patient_ID', 'Location', 'Age', 'Sex', 'Laterality', 'Primary', 'Hospital', 'Internal_test_set', 'Unnamed: 0', 'Expert1', 'Expert2', 'Expert3', 'Image_Name_y', 'Pathologic_Fracture', 'Model']
    expert_df['Consensus'] = expert_df[['Expert1', 'Expert2', 'Expert3']].apply(majority_rule, axis=1)
    expert_df_only_expert = expert_df.loc[:, ['Expert1', 'Expert2', 'Expert3']]
    expert_df = expert_df.loc[:, ['Image_Name_x', 'Probability', 'Class', 'Patient_ID', 'Location', 'Age',
       'Sex', 'Laterality', 'Primary', 'Hospital', 'Internal_test_set',
        'Expert1', 'Expert2', 'Expert3', 'Consensus']]
    expert_df.columns = ['Image_Name', 'Probability', 'Class', 'Patient_ID', 'Location', 'Age',
       'Sex', 'Laterality', 'Primary', 'Hospital', 'Internal_test_set',
        'Expert1', 'Expert2', 'Expert3', 'Consensus']
    expert_df = pd.merge(expert_df, sample_df, how = 'left', on = 'Image_Name')
    expert_df.to_csv(os.path.join(snuh_test_save_dir, 'result_with_expert.csv'), index = False)

    expert_array_only_expert = np.array(expert_df_only_expert)
    category_counts = np.array([np.bincount(row, minlength=2) for row in expert_array_only_expert])
    # Compute Fleiss' Kappa
    kappa = fleiss_kappa(category_counts, method='fleiss')
    print("Fleiss' Kappa score: " + str(kappa))

    target_list = ['Expert1', 'Expert2', 'Expert3', 'Consensus', 'Model']
    for target in target_list:
        print(target)
        y_true = expert_df['Class']
        y_pred = expert_df[target]

        result_dict = calculate_metrics(y_true, y_pred)

        ci_dict = bootstrap_ci(y_true, y_pred)
        for key in ci_dict.keys():
            print(f"{key}: {result_dict[key]:.2f} ({ci_dict[key]['lower_bound']:.2f}-{ci_dict[key]['upper_bound']:.2f})")

        if target != 'Model':
            p_value = mcnemar_test(y_true, y_pred, expert_df['Model'])
            print(f"P-value for comparison between {target} and Model: {p_value}")