import torch.utils.data
import torch
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve
import itertools
import seaborn as sns
from scipy.stats import ttest_rel
from statannot import add_stat_annotation
from itertools import combinations
import json
from tqdm import tqdm
import matplotlib
import pickle
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def calculate_probability(model, data, device, train_mode, meta_df):
    image_name_list = []
    probability_list = []
    for i in range(len(data)):
        original_image, fracture_image, target, original_image_original, fracture_image_original, image_name = data[i]
        image_name_list.append(image_name)
        if train_mode == 'Original':
            inputs = original_image.to(device)
            inputs = inputs.unsqueeze(0)
            outputs = model(inputs)
        elif train_mode == 'Fracture':
            inputs = fracture_image.to(device)
            inputs = inputs.unsqueeze(0)
            outputs = model(inputs)
        elif train_mode == 'Dual':
            original_inputs = original_image.to(device)
            fracture_inputs = fracture_image.to(device)
            original_inputs = original_inputs.unsqueeze(0)
            fracture_inputs = fracture_inputs.unsqueeze(0)
            outputs = model(original_inputs, fracture_inputs)
        outputs = torch.sigmoid(outputs)
        probability = outputs[0][1].item()
        probability_list.append(probability)

    result_df = pd.DataFrame(list(zip(image_name_list, probability_list)), columns=['Image_Name', 'Probability'])
    result_df = pd.merge(result_df, meta_df, how='left', on='Image_Name')

    return result_df

def calculate_metrics_crossvalidation(df, name, fold, train_mode):
    auroc_score = roc_auc_score(df['Class'], df['Probability'])
    precision, recall, _ = precision_recall_curve(df['Class'], df['Probability'])
    auprc_score = auc(recall, precision)

    result_dict = {'hospital': name, 'train_mode': train_mode, 'fold': fold, 'AUROC': auroc_score, 'AUPRC': auprc_score}

    return result_dict

def draw_boxplot_and_compare(df, hospital, comparison_param, metric, fig_dir, significance_level=0.05):
    print(hospital)
    # Filter the DataFrame based on the dataset
    filtered_df = df[df['hospital'] == hospital].copy()
    # filtered_df['train_mode'] = filtered_df['train_mode'] + '_' + filtered_df['train_location']
    filtered_df['train_mode'] = filtered_df['train_mode'].replace({'original': 'Original', 'fracture': 'Fracture', 'dual': 'Dual'})

    # Plotting
    plt.figure(figsize=(7, 6))
    boxplot = sns.boxplot(x=comparison_param, y=metric, data=filtered_df, palette="Set3", linewidth=0.7)
    boxplot.set_xlabel('')
    # plt.title(f"Boxplot of {metric} for {comparison_param} within {hospital} and {location}")

    # Identifying all unique groups
    groups = filtered_df[comparison_param].unique()

    # Conducting Wilcoxon signed-rank test for all pairs
    pairs = list(combinations(groups, 2))
    significant_pairs = []
    for pair in pairs:
        group1_data = filtered_df[filtered_df[comparison_param] == pair[0]][metric]
        group2_data = filtered_df[filtered_df[comparison_param] == pair[1]][metric]

        # Ensure both groups have data and equal lengths
        min_length = min(len(group1_data), len(group2_data))
        if min_length > 0:
            stat, p_value = ttest_rel(group1_data[:min_length], group2_data[:min_length])
            # p_value = bonferroni(p_value, min_length)
            if p_value < significance_level:
                significant_pairs.append(pair)
    # Using statannot to add annotations
    if len(significant_pairs) > 0:
        test_results = add_stat_annotation(
            boxplot, data=filtered_df, x=comparison_param, y=metric,
            box_pairs=significant_pairs, test='t-test_paired', text_format='star', loc='inside', verbose=2,
            comparisons_correction=None, line_offset=0.1, line_offset_to_box=0.1, line_height=0.03, text_offset=0.05,
            linewidth=0.7
            # Adjust these parameters
        )

    plt.subplots_adjust(top=0.9)
    plt.ylim([0.5, 1.0])
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f"Boxplot of {metric} for {comparison_param} within {hospital}.pdf"),
        transparent=True)
    plt.savefig(
        os.path.join(fig_dir, f"Boxplot of {metric} for {comparison_param} within {hospital}.eps"),
        format='eps', transparent=True)
    plt.clf()
    plt.close()
    plt.cla()

def plot_roc_curve(y_true, y_scores, save_dir, name, color='darkorange'):
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plotting ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, f'{name}_ROC_Curve.png'))
    plt.savefig(os.path.join(save_dir, f'{name}_ROC_Curve.pdf'), transparent=True)
    plt.clf()
    plt.close()

    return roc_auc, fpr, tpr, thresholds

def plot_pr_curve_and_select_threshold(y_true, y_scores, save_dir, name, color='darkorange'):
    # Calculate Precision-Recall curve and AUC
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    target_criterion = (2.0 * tpr) - fpr
    optimal_idx = np.argmax(target_criterion)
    best_threshold = thresholds[optimal_idx]

    # Plotting PR curve
    plt.figure()
    plt.plot(recall, precision, color=color, lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_dir, f'{name}_PR_Curve.png'))
    plt.savefig(os.path.join(save_dir, f'{name}_PR_Curve.pdf'), transparent=True)
    plt.clf()
    plt.close()

    return best_threshold

def draw_roc_curve(save_dir, name, color):
    save_dir = os.path.join(save_dir, name)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    sample_df = pd.read_csv(os.path.join(save_dir, name + '_result.csv'))

    y_true = sample_df['Class'].to_list()
    y_scores = sample_df['Probability'].to_list()

    auc, fpr, tpr, thresholds = plot_roc_curve(y_true, y_scores, save_dir, name, color)
    threshold = plot_pr_curve_and_select_threshold(y_true, y_scores, save_dir, name, color)
    roc_metric = {}
    roc_metric['FPR'] = fpr
    roc_metric['TPR'] = tpr
    roc_metric['Thresholds'] = thresholds

    with open(os.path.join(save_dir, 'ROC_metric.pickle'), 'wb') as f:
        pickle.dump(roc_metric, f, pickle.HIGHEST_PROTOCOL)

    return threshold

def draw_roc_curve_all(save_dir, name_target_list, colors):
    name_list = os.listdir(save_dir)
    name_list = [name for name in name_list if name in name_target_list]

    final_df = pd.DataFrame()
    # Initialize plot
    plt.figure(figsize=(10, 8))
    for name in name_list:
        save_dir_name = os.path.join(save_dir, name)
        if os.path.exists(save_dir_name) is False:
            os.mkdir(save_dir_name)

        sample_df = pd.read_csv(os.path.join(save_dir_name, name + '_result.csv'))

        if len(final_df) == 0:
            final_df = sample_df
        else:
            final_df = pd.concat((final_df, sample_df))

        # Generating synthetic data
        np.random.seed(0)
        y_true = sample_df['Class'].to_list()
        y_scores = sample_df['Probability'].to_list()

        # Iterate over each model's scores
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, color=colors[name], label=f'{name} (area = {roc_auc:.3f})')

    name = 'Whole_test'
    y_true = final_df['Class'].to_list()
    y_scores = final_df['Probability'].to_list()

    # Iterate over each model's scores
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, color=colors[name], label=f'{name} (area = {roc_auc:.3f})')

    # Plotting diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Adding labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(save_dir, 'ROC_total.png'))
    plt.savefig(os.path.join(save_dir, 'ROC_total.pdf'), transparent=True)

    plt.clf()
    plt.close()
    plt.cla()

def draw_pr_curve_all(save_dir, name_target_list, colors):
    name_list = os.listdir(save_dir)
    name_list = [name for name in name_list if name in name_target_list]
    # Initialize plot
    final_df = pd.DataFrame()
    plt.figure(figsize=(10, 8))
    for name in name_list:
        save_dir_name = os.path.join(save_dir, name)
        if os.path.exists(save_dir_name) is False:
            os.mkdir(save_dir_name)

        sample_df = pd.read_csv(os.path.join(save_dir_name, name + '_result.csv'))
        if len(final_df) == 0:
            final_df = sample_df
        else:
            final_df = pd.concat(((final_df, sample_df)))
        # Generating synthetic data
        np.random.seed(0)
        y_true = sample_df['Class'].to_list()
        y_scores = sample_df['Probability'].to_list()

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, color=colors[name], label=f'{name} PR (area = {pr_auc:.3f})')

    name = 'Whole_test'
    y_true = final_df['Class'].to_list()
    y_scores = final_df['Probability'].to_list()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, lw=2, color=colors[name], label=f'{name} PR (area = {pr_auc:.3f})')

    # Adding labels and title

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curves')
    plt.legend(loc="lower left")

    plt.savefig(os.path.join(save_dir, 'PRC_total.png'))
    plt.savefig(os.path.join(save_dir, 'PRC_total.pdf'), transparent=True)
    plt.clf()
    plt.close()
    plt.cla()

def plot_confusion_matrix(y_true, y_pred, save_dir, name, color):
    # Calculating confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cmap = sns.light_palette(color, as_cmap=True)

    # Plotting confusion matrix
    plt.figure(figsize=(8, 6))
    # plt.imshow(cm, interpolation='nearest', cmap='cmap')
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.clim(0.0, 1.0)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Standard', 'Pathologic'])
    plt.yticks(tick_marks, ['Standard', 'Positive'])

    # Adding text annotations
    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(save_dir, f'{name}_confusion_matrix.png'))
    plt.savefig(os.path.join(save_dir, f'{name}_confusion_matrix.pdf'), transparent=True)
    plt.clf()
    plt.close()

def draw_confusion_matrix(save_dir, name, threshold, color):
    save_dir = os.path.join(save_dir, name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sample_df = pd.read_csv(os.path.join(save_dir, name + '_result.csv'))
    y_true = sample_df['Class'].to_list()
    y_scores = sample_df['Probability'].to_list()

    # Convert lists to numpy arrays for consistency
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Apply threshold to generate binary predictions
    y_pred = (y_scores > threshold).astype(int)

    auroc, _, _, _ = plot_roc_curve(y_true, y_scores, save_dir, name)
    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    plot_confusion_matrix(y_true, y_pred, save_dir, name, color)
    # Calculate metrics
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    sensitivity = tp / (tp + fn)  # Recall
    specificity = tn / (tn + fp)
    ppv = precision_score(y_true, y_pred)  # Positive Predictive Value (Precision)
    npv = tn / (tn + fn)  # Negative Predictive Value
    f1 = f1_score(y_true, y_pred)  # F1 Score
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)

    # Optionally, print the metrics or save them to a file
    metrics = {
        'AUC': auroc,
        'mAP': auprc,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'F1 Score': f1
    }

    return metrics

def draw_confusion_matrix_prob_distribution(save_dir, name_target_list, threshold, color):
    name_list = os.listdir(save_dir)
    name_list = [name for name in name_list if name in name_target_list]
    final_df = pd.DataFrame()
    plt.figure(figsize=(10, 8))
    for name in name_list:
        save_dir_name = os.path.join(save_dir, name)
        if os.path.exists(save_dir_name) is False:
            os.mkdir(save_dir_name)

        sample_df = pd.read_csv(os.path.join(save_dir_name, name + '_result.csv'))
        if len(final_df) == 0:
            final_df = sample_df
        else:
            final_df = pd.concat(((final_df, sample_df)))

    name = 'Whole_test'
    y_true = final_df['Class'].to_list()
    y_scores = final_df['Probability'].to_list()

    # Convert lists to numpy arrays for consistency
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Apply threshold to generate binary predictions
    y_pred = (y_scores > threshold).astype(int)

    plot_confusion_matrix(y_true, y_pred, save_dir, name, color)

    # Separate data into two classes
    class_0_df = final_df[final_df['Class'] == 0]
    class_1_df = final_df[final_df['Class'] == 1]

    class_0 = class_0_df['Probability'].to_list()
    class_1 = class_1_df['Probability'].to_list()

    plt.figure(figsize=(10, 6))

    # Normalize counts
    n_class_0, bins_class_0 = np.histogram(class_0, bins=40)
    n_class_1, bins_class_1 = np.histogram(class_1, bins=40)

    n_class_0 = n_class_0 / (n_class_0.sum())
    n_class_1 = n_class_1 / (n_class_1.sum())

    bin_centers_class_0 = 0.5 * (bins_class_0[1:] + bins_class_0[:-1])
    bin_centers_class_1 = 0.5 * (bins_class_1[1:] + bins_class_1[:-1])

    plt.bar(bin_centers_class_0, n_class_0, width=(bins_class_0[1] - bins_class_0[0]), alpha=0.5, label='Class 0')
    plt.bar(bin_centers_class_1, n_class_1, width=(bins_class_1[1] - bins_class_1[0]), alpha=0.5, label='Class 1')

    plt.xlabel('Score')
    plt.ylabel('Frequency (normalized)')
    plt.ylim(0, 0.2)  # Set y-axis range from 0 to 1
    plt.legend(loc='upper right')
    plt.title('Score Reference Plot')
    plt.plot([float(threshold), float(threshold)], [0, 0.2], color='navy', lw=2, linestyle='--')
    plt.savefig(os.path.join(save_dir, f'{name}_distribution.png'))
    plt.savefig(os.path.join(save_dir, f'{name}_distribution.pdf'), transparent=True)

def majority_rule(row):
    counts = row.value_counts()
    max_count = counts.max()
    if (counts == max_count).sum() == 1:
        return counts.idxmax()
    else:
        return np.nan  # Handle ties as needed

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    ppv = precision_score(y_true, y_pred)
    npv = tn / (tn + fn)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    result_dict = {'accuracy': accuracy,
                   'F1 score': f1,
                   'Sensitivity': sensitivity,
                   'Specificity': specificity,
                   'PPV': ppv,
                   'NPV': npv}

    return result_dict

def bootstrap_ci(y_true, y_pred, n_bootstrap=1000):
    bootstrap_dict = {}
    ci_dict = {}
    for _ in range(n_bootstrap):
        # Resample (with replacement) the indices
        indices = resample(np.arange(len(y_true)), replace=True)
        result_dict = calculate_metrics(y_true[indices], y_pred[indices])
        for key in result_dict.keys():
            if key in bootstrap_dict.keys():
                bootstrap_dict[key].append(result_dict[key])
            else:
                bootstrap_dict[key] = [result_dict[key]]

    for key in bootstrap_dict.keys():
        lower_bound = np.percentile(bootstrap_dict[key], 2.5)
        upper_bound = np.percentile(bootstrap_dict[key], 97.5)
        ci_dict[key] = {'lower_bound': lower_bound, 'upper_bound': upper_bound}

    return ci_dict

def mcnemar_test(y_true, y_pred1, y_pred2):
    b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    table = np.array([[0, b], [c, 0]])
    result = mcnemar(table, exact=True)  # Use exact=True if sample size is small
    return result.pvalue


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
                 discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        return rollout(self.attentions, self.discard_ratio, self.head_fusion), self.attentions


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), dim=-1, largest=False)

            flat[0, indices] = 0
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

        mask = result[0, 0, 1:]
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)

        return mask


def show_mask_on_image_global(img, mask):
    img = np.float32(img) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    alpha = 0.4
    overlay = (1 - alpha) * img + alpha * heatmap

    return np.uint8(255 * overlay)


def show_cam_on_image(img, mask):
    img = np.float32(img) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def show_cam_on_image_fracture(img, mask, selected_bbox):
    img = np.float32(img) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = np.float32(img)
    cam[int(selected_bbox[0][1]):int(selected_bbox[1][1]), int(selected_bbox[0][0]):int(selected_bbox[1][0]),
    :] += heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def new_attention_analysis_original(model, save_dir, data, device, fusion_method, discard_ratio):
    attention_rollout = VITAttentionRollout(model, head_fusion=fusion_method, discard_ratio=discard_ratio)
    save_dir_pathologic = os.path.join(save_dir, 'pathologic')
    if os.path.exists(save_dir_pathologic) is False:
        os.mkdir(save_dir_pathologic)
    save_dir_standard = os.path.join(save_dir, 'standard')
    if os.path.exists(save_dir_standard) is False:
        os.mkdir(save_dir_standard)
    save_dir_pathologic_overall = os.path.join(save_dir_pathologic, 'overall')
    if os.path.exists(save_dir_pathologic_overall) is False:
        os.mkdir(save_dir_pathologic_overall)
    save_dir_standard_overall = os.path.join(save_dir_standard, 'overall')
    if os.path.exists(save_dir_standard_overall) is False:
        os.mkdir(save_dir_standard_overall)

    with tqdm(total=len(data)) as pbar:
        for i in range(len(data)):
            original_img, fracture_img, label, original_image_original, fracture_image_original, image_name = data[i]
            original_inputs = original_img.to(device)
            original_inputs = original_inputs.unsqueeze(0)
            with torch.no_grad():
                outputs = model(original_inputs)
            outputs = torch.sigmoid(outputs).detach().cpu()
            label = label[1].item()
            probability = outputs[0][1].item()

            mask_original, attentions_original = attention_rollout(original_inputs)
            del original_inputs

            original_image_original_resized = original_image_original.resize((224, 224))
            normalized_attention_map_original = (mask_original - np.min(mask_original)) / (
                    np.max(mask_original) - np.min(mask_original))
            resized_attention_map_original = cv2.resize(normalized_attention_map_original, (224, 224),
                                                        interpolation=cv2.INTER_LINEAR)
            np_img_original = np.array(original_image_original_resized)[:, :, ::-1]
            attention_mask_original = show_cam_on_image(np_img_original, resized_attention_map_original)
            attention_mask_original = cv2.resize(attention_mask_original, original_image_original.size)
            if label == 1:
                cv2.imwrite(os.path.join(save_dir_pathologic_overall,
                                         image_name + '_' + str(label) + '_' + str(probability) + '.png'),
                            attention_mask_original)
            else:
                cv2.imwrite(os.path.join(save_dir_standard_overall,
                                         image_name + '_' + str(label) + '_' + str(probability) + '.png'),
                            attention_mask_original)
            pbar.update()

def new_attention_analysis_fracture(model, save_dir, data, device, fusion_method, discard_ratio, json_dir):
    attention_rollout = VITAttentionRollout(model, head_fusion=fusion_method, discard_ratio=discard_ratio)
    save_dir_pathologic = os.path.join(save_dir, 'pathologic')
    if os.path.exists(save_dir_pathologic) is False:
        os.mkdir(save_dir_pathologic)
    save_dir_standard = os.path.join(save_dir, 'standard')
    if os.path.exists(save_dir_standard) is False:
        os.mkdir(save_dir_standard)
    save_dir_pathologic_overall = os.path.join(save_dir_pathologic, 'overall')
    if os.path.exists(save_dir_pathologic_overall) is False:
        os.mkdir(save_dir_pathologic_overall)
    save_dir_standard_overall = os.path.join(save_dir_standard, 'overall')
    if os.path.exists(save_dir_standard_overall) is False:
        os.mkdir(save_dir_standard_overall)


    with tqdm(total=len(data)) as pbar:
        for i in range(len(data)):
            original_img, fracture_img, label, original_image_original, fracture_image_original, image_name = data[i]
            inputs = fracture_img.to(device)
            inputs = inputs.unsqueeze(0)
            with torch.no_grad():
                outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            label = label[1].item()
            probability = outputs[0][1].item()
            sample = image_name.split('.tif')[0]
            json_file = os.path.join(json_dir, sample + '.json')
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            fracture_data = json_data['shapes'][1]
            if fracture_data['label'] != 'Fracture':
                fracture_data = json_data['shapes'][0]
            selected_bbox = fracture_data['points']
            mask, attentions = attention_rollout(inputs)
            normalized_attention_map = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
            bbox_width = int(selected_bbox[1][0]) - int(selected_bbox[0][0])
            bbox_height = int(selected_bbox[1][1]) - int(selected_bbox[0][1])
            resized_attention_map = cv2.resize(normalized_attention_map, (int(bbox_width), int(bbox_height)),
                                               interpolation=cv2.INTER_LINEAR)
            np_img = np.array(original_image_original)[:, :, ::-1]
            attention_mask = show_cam_on_image_fracture(np_img, resized_attention_map, selected_bbox)
            if label == 1:
                cv2.imwrite(os.path.join(save_dir_pathologic_overall,
                                         image_name + '_' + str(label) + '_' + str(probability) + '.png'),
                            attention_mask)
            else:
                cv2.imwrite(os.path.join(save_dir_standard_overall,
                                         image_name + '_' + str(label) + '_' + str(probability) + '.png'),
                            attention_mask)


            pbar.update()

