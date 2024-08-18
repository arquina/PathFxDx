import argparse
import sys
import torch
import os
import numpy as np
import timm
import pandas as pd

from interpretation_utils import calculate_probability, calculate_metrics_crossvalidation
from interpretation_utils import draw_boxplot_and_compare

def Parser_main():
    parser = argparse.ArgumentParser(description="Bone fracture classification")
    parser.add_argument("--rootdir" , default = "/home/seob/class_project/20231106_pathologic/" , help="dataset rootdir", type = str)
    parser.add_argument("--script_dir", default='/home/seob/PathFxDx/script/', type=str)
    parser.add_argument("--cuda", default = 'cuda:1', type = str, help = 'cuda device or cpu')
    parser.add_argument("--random_seed", default = 1234567, type = int)
    parser.add_argument("--analysis_dir", default = "/home/seob/PathFxDx/result/240523_final_data_final_weight1.5/5fold_comparison/", type = str)
    parser.add_argument("--Original_analysis_dir", default = "2024-05-23_10:55:06_epoch_100_lr_0.001_decay_0.001_box_size_None", help = "result save directory", type = str)
    parser.add_argument("--Fracture_analysis_dir", default = "2024-05-23_21:00:53_epoch_100_lr_0.001_decay_0.001_box_size_None", help = "result save directory", type = str)
    parser.add_argument("--Dual_analysis_dir", default = "2024-05-23_12:24:23_epoch_100_lr_0.001_decay_0.001_box_size_None", help = "result save directory", type = str)
    parser.add_argument("--FF_number", default = 5, type = int)
    parser.add_argument('--clinical_data', default = "/home/seob/class_project/Dataset_20230822.xlsx", help = 'clinical_data')
    parser.add_argument('--bbox_size', default = None, type = int)
    parser.add_argument('--target_metric', default = 'AUPRC', type = str, help = 'Target metric for select best model')
    parser.add_argument('--probability_calculation', default = False, action = 'store_true', help = 'Probability calculation takes long, if it is already performed set this False')
    return parser.parse_args()

Argument = Parser_main()
sys.path.append(Argument.script_dir)
from Dataloader import xray_dataset_class_val
from model import CustomViT

clinical_data = pd.read_excel(Argument.clinical_data)
data_root_dir = Argument.rootdir
root_dir = Argument.analysis_dir
target_metric = Argument.target_metric
probability_calculation = Argument.probability_calculation

dataset_result_df_list = []
dataset_list = ['original', 'fracture', 'Dual']
for dataset in dataset_list:
    result_dict_list = []
    final_class_list = []
    dataset_root_dir = os.path.join(root_dir, dataset, 'All')
    if dataset == 'original':
        fold_root_dir = os.path.join(dataset_root_dir, Argument.Original_analysis_dir)
    elif dataset == 'fracture':
        fold_root_dir = os.path.join(dataset_root_dir, Argument.Fracture_analysis_dir)
    elif dataset == 'Dual':
        fold_root_dir = os.path.join(dataset_root_dir, Argument.Dual_analysis_dir)

    save_dir = os.path.join(fold_root_dir, 'analyze')
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    meta_df = clinical_data.copy()
    save_dir = os.path.join(save_dir, target_metric)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    snuh_df = meta_df[meta_df['Hospital'] == 'SNUH']
    snuh_test_df = snuh_df[snuh_df['Internal_test_set'] == 1]
    snubh_df = meta_df[meta_df['Hospital'] == 'SNUBH']
    ncc_df = meta_df[meta_df['Hospital'] == 'NCC']
    vincent_df = meta_df[meta_df['Hospital'] == 'Vincent']
    fold_list = ['0', '1', '2', '3', '4']
    for fold in fold_list:
        print(fold)
        fold_dir = os.path.join(fold_root_dir, fold)
        if probability_calculation:
            model_path_list = os.listdir(os.path.join(fold_dir, 'model'))
            print(model_path_list)
            if target_metric == 'AUROC':
                metric_list = [float(model_path.split('_')[1]) for model_path in model_path_list]
            else:
                metric_list = [float(model_path.split('_')[3]) for model_path in model_path_list]
            target_model_idx = np.argmax(np.array(metric_list))
            model_path = model_path_list[target_model_idx]
            model_absolute_path = os.path.join(fold_dir, 'model', model_path)

            snuh_val_df_path = os.path.join(fold_dir, 'patient_val_set.csv')
            snuh_val_df = pd.read_csv(snuh_val_df_path)

            print("Load classification dataset")

            snuh_val_data = xray_dataset_class_val(snuh_val_df, data_root_dir)
            snuh_test_data = xray_dataset_class_val(snuh_test_df, data_root_dir)
            snubh_data = xray_dataset_class_val(snubh_df, data_root_dir)
            ncc_data = xray_dataset_class_val(ncc_df, data_root_dir)
            vincent_data = xray_dataset_class_val(vincent_df, data_root_dir)

            print("Load classification model")
            if dataset == 'Dual':  # number of unique locations
                model = CustomViT()
            else:
                model = timm.create_model('vit_base_patch16_224_dino', pretrained=True, num_classes=2)

            device = torch.device(Argument.cuda)
            state_dict = torch.load(model_absolute_path, map_location = device)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            print("Loading model complete")

            print('Calculate Probability')
            snuh_val_result_df = calculate_probability(model, snuh_val_data, device, dataset, meta_df)
            snuh_test_result_df = calculate_probability(model, snuh_test_data, device, dataset, meta_df)
            snubh_result_df = calculate_probability(model, snubh_data, device, dataset, meta_df)
            ncc_result_df = calculate_probability(model, ncc_data, device, dataset, meta_df)
            vincent_result_df = calculate_probability(model, vincent_data, device, dataset, meta_df)

            probability_df = pd.concat((snuh_val_result_df, snuh_test_result_df, snubh_result_df, ncc_result_df, vincent_result_df)).reset_index(drop=True)
            probability_df.to_csv(os.path.join(fold_dir, 'Probability_result.csv'), index=False)
        else:
            probability_df = pd.read_csv(os.path.join(fold_dir, 'Probability_result.csv'))

        probability_df_external = probability_df[probability_df['Hospital'] != 'SNUH']
        probability_df_external = probability_df[probability_df['Hospital'] != 'BRMH']

        probability_df_snuh = probability_df[probability_df['Hospital'] == 'SNUH']
        probability_df_snuh_val = probability_df_snuh[probability_df_snuh['Internal_test_set'] == 0]
        probability_df_snuh_test = probability_df_snuh[probability_df_snuh['Internal_test_set'] == 1]
        probability_df_snubh = probability_df[probability_df['Hospital'] == 'SNUBH']
        probability_df_ncc = probability_df[probability_df['Hospital'] == 'NCC']
        probability_df_vincent = probability_df[probability_df['Hospital'] == 'Vincent']

        result_dict = calculate_metrics_crossvalidation(probability_df, 'total', fold, dataset)
        result_dict_list.append(result_dict)
        result_dict = calculate_metrics_crossvalidation(probability_df_snuh_val, 'SNUH_val', fold, dataset)
        result_dict_list.append(result_dict)
        result_dict = calculate_metrics_crossvalidation(probability_df_snuh_test, 'SNUH_test', fold, dataset)
        result_dict_list.append(result_dict)
        result_dict = calculate_metrics_crossvalidation(probability_df_external, 'External', fold, dataset)
        result_dict_list.append(result_dict)
        result_dict = calculate_metrics_crossvalidation(probability_df_snubh, 'SNUBH', fold, dataset)
        result_dict_list.append(result_dict)
        result_dict = calculate_metrics_crossvalidation(probability_df_ncc, 'NCC', fold, dataset)
        result_dict_list.append(result_dict)
        result_dict = calculate_metrics_crossvalidation(probability_df_vincent, 'SVH', fold, dataset)
        result_dict_list.append(result_dict)

    result_df = pd.DataFrame(result_dict_list)
    result_df.to_csv(os.path.join(save_dir, 'Best' + target_metric + '_final_result.csv'), index=False)

    hospital_list = list(set(result_df['hospital'].to_list()))
    metric_data_list = ['AUROC', 'AUPRC']
    for hospital in hospital_list:
        class_result_df = result_df[result_df['hospital'] == hospital]
        class_dict = {}
        class_dict['hospital'] = hospital
        for metric_data in metric_data_list:
            class_metric_data = class_result_df[metric_data].to_list()
            class_metric_mean_value = np.mean(class_metric_data)
            class_metric_std = np.std(class_metric_data, ddof = 0)
            class_dict[metric_data + '_mean'] = class_metric_mean_value
            class_dict[metric_data + '_std'] = class_metric_std
        final_class_list.append(class_dict)

    class_df = pd.DataFrame(final_class_list)
    class_df.to_csv(os.path.join(save_dir, 'Best' + target_metric + '_final_stat.csv'), index = False)
    dataset_result_df_list.append(result_df)

fig_dir = os.path.join(root_dir, 'figure')
if os.path.exists(fig_dir) is False:
    os.mkdir(fig_dir)
fig_dir = os.path.join(fig_dir, target_metric)
if os.path.exists(fig_dir) is False:
    os.mkdir(fig_dir)

final_result_df = pd.concat(dataset_result_df_list)
draw_boxplot_and_compare(final_result_df, 'total', 'train_mode', 'AUROC', fig_dir)
draw_boxplot_and_compare(final_result_df, 'External', 'train_mode', 'AUROC', fig_dir)
draw_boxplot_and_compare(final_result_df, 'SNUBH', 'train_mode', 'AUROC', fig_dir)
draw_boxplot_and_compare(final_result_df, 'SNUH_test', 'train_mode', 'AUROC', fig_dir)
draw_boxplot_and_compare(final_result_df, 'SNUH_val', 'train_mode', 'AUROC', fig_dir)
draw_boxplot_and_compare(final_result_df, 'NCC', 'train_mode', 'AUROC', fig_dir)
draw_boxplot_and_compare(final_result_df, 'SVH', 'train_mode', 'AUROC', fig_dir)

draw_boxplot_and_compare(final_result_df, 'total', 'train_mode', 'AUPRC', fig_dir)
draw_boxplot_and_compare(final_result_df, 'External', 'train_mode', 'AUPRC', fig_dir)
draw_boxplot_and_compare(final_result_df, 'SNUBH', 'train_mode', 'AUPRC', fig_dir)
draw_boxplot_and_compare(final_result_df, 'SNUH_test', 'train_mode','AUPRC', fig_dir)
draw_boxplot_and_compare(final_result_df, 'SNUH_val', 'train_mode', 'AUPRC', fig_dir)
draw_boxplot_and_compare(final_result_df, 'NCC', 'train_mode', 'AUPRC', fig_dir)
draw_boxplot_and_compare(final_result_df, 'SVH', 'train_mode', 'AUPRC', fig_dir)

