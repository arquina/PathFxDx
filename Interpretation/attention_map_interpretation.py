import argparse
import torch
import os
import pandas as pd
from interpretation_utils import  new_attention_analysis_original, new_attention_analysis_fracture
import numpy as np
import timm

def Parser_main():
    parser = argparse.ArgumentParser(description="Bone fracture classification")
    parser.add_argument("--rootdir" , help="dataset rootdir", type = str)
    parser.add_argument("--script_dir", help = 'Location of the PathFxDx scripts rootdir', type = str)
    parser.add_argument("--cuda", default = 'cuda:0', type = str, help = 'cuda device or cpu')
    parser.add_argument("--random_seed", default = 1234567, type = int)
    parser.add_argument("--dataset", default = 'Original', type = str)
    parser.add_argument("--analysis_dir", type = str)
    parser.add_argument("--model_dir", help = "result save directory", type = str)
    parser.add_argument("--model_name", default = None, help = 'model name in model dir', type = str)
    parser.add_argument('--clinical_data', help = 'clinical_data')
    parser.add_argument('--bbox_size', default = None, type = int)
    parser.add_argument('--target_metric', default = 'AUPRC', type = str, help = 'Target metric for select best model')
    return parser.parse_args()

import sys
Argument = Parser_main()
sys.path.append(Argument.script_dir)
from Dataloader import xray_dataset_class_val
from model import CustomViT

meta_df = pd.read_excel(Argument.clinical_data)

dataset = Argument.dataset
model_path_dir = Argument.model_dir
target_metric = Argument.target_metric
model_dir = os.path.join(Argument.analysis_dir, dataset, 'All', Argument.model_dir)
if Argument.model_name == None:
    model_path_list = os.listdir(os.path.join(model_dir, 'model'))
    if target_metric == 'AUROC':
        metric_list = [float(model_path.split('_')[2]) for model_path in model_path_list]
    elif target_metric == 'AUPRC':
        metric_list = [float(model_path.split('_')[5]) for model_path in model_path_list]
    target_model_idx = np.argmax(np.array(metric_list))
    model_path = model_path_list[target_model_idx]
else:
    model_path = Argument.model_name

model_absolute_path = os.path.join(model_dir, 'model', model_path)
save_dir = os.path.join(Argument.analysis_dir, dataset, 'All', Argument.model_dir, 'analyze')
attention_map_dir = os.path.join(save_dir, 'attention_map')
if os.path.exists(attention_map_dir) is False:
    os.mkdir(attention_map_dir)

snuh_df = meta_df[meta_df['Hospital'] == 'SNUH']
snuh_train_df = snuh_df[snuh_df['Internal_test_set'] == 0]
snuh_test_df = snuh_df[snuh_df['Internal_test_set'] == 1]
external_df = meta_df[meta_df['Hospital'] != 'SNUH']
external_df = external_df[external_df['Hospital'] != 'BRMH']

print("Load classification dataset")
root_dir = Argument.rootdir
snuh_train_data = xray_dataset_class_val(snuh_train_df, root_dir)
snuh_test_data = xray_dataset_class_val(snuh_test_df, root_dir)
external_data = xray_dataset_class_val(external_df, root_dir)
print("Load classification model")

if dataset == 'Dual':
    model = CustomViT()
else:
    model = timm.create_model('vit_base_patch16_224_dino', pretrained=True, num_classes=2)

state_dict = torch.load(model_absolute_path)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
device = torch.device(Argument.cuda)
model.to(device)
print("Loading model complete")

save_dir_snuh = os.path.join(attention_map_dir, 'snuh')
if os.path.exists(save_dir_snuh) is False:
    os.mkdir(save_dir_snuh)

save_dir_external = os.path.join(attention_map_dir, 'external')
if os.path.exists(save_dir_external) is False:
    os.mkdir(save_dir_external)

save_dir_snuh_train = os.path.join(save_dir_snuh, 'train')
if os.path.exists(save_dir_snuh_train) is False:
    os.mkdir(save_dir_snuh_train)

save_dir_snuh_test = os.path.join(save_dir_snuh, 'test')
if os.path.exists(save_dir_snuh_test) is False:
    os.mkdir(save_dir_snuh_test)

fusion_method = 'min'
discard_ratio = 0


if dataset == 'Original':
    new_attention_analysis_original(model, save_dir_snuh_train, snuh_train_data, device, fusion_method, discard_ratio)
    new_attention_analysis_original(model, save_dir_snuh_test, snuh_test_data, device, fusion_method, discard_ratio)
    new_attention_analysis_original(model, save_dir_external, external_data, device, fusion_method, discard_ratio)
elif dataset == 'Fracture':
    new_attention_analysis_fracture(model, save_dir_snuh_train, snuh_train_data, device, fusion_method, discard_ratio, Argument.rootdir)
    new_attention_analysis_fracture(model, save_dir_snuh_test, snuh_test_data, device, fusion_method, discard_ratio, Argument.rootdir)
    new_attention_analysis_fracture(model, save_dir_external, external_data, device, fusion_method, discard_ratio, Argument.rootdir)