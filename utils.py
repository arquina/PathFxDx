import numpy as np
import torch
from sklearn import metrics
import pandas as pd

def calculate_class_balanced_weights(subclass_counts, beta=0.999):
    effective_num = 1.0 - np.power(beta, subclass_counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(subclass_counts)
    return weights

def get_class_balanced_weight(clinical_data):
    clinical_data['location_encoded'] = pd.factorize(clinical_data['Location'])[0]
    clinical_data_sub = clinical_data[clinical_data['Location'] == 'Subtrochanteric area']
    clinical_data_inter = clinical_data[clinical_data['Location'] == 'Intertrochanteric area']
    clinical_data_neck = clinical_data[clinical_data['Location'] == 'Neck']
    clinical_data_lt = clinical_data[clinical_data['Location'] == 'Isolated LT']
    clinical_data_gt = clinical_data[clinical_data['Location'] == 'Isolated GT']

    subclass_weight = calculate_class_balanced_weights(
        subclass_counts=[len(clinical_data_sub[clinical_data_sub['Class'] == 0]),
                         len(clinical_data_sub[clinical_data_sub['Class'] == 1]),
                         len(clinical_data_neck[clinical_data_neck['Class'] == 0]),
                         len(clinical_data_neck[clinical_data_neck['Class'] == 1]),
                         len(clinical_data_inter[clinical_data_inter['Class'] == 0]),
                         len(clinical_data_inter[clinical_data_inter['Class'] == 1]),
                         1.0, # Isolated LT class has no negative label so it was set to 1.0 (There are no data so this weight will not be used)
                         len(clinical_data_lt[clinical_data_lt['Class'] == 1]),
                         len(clinical_data_gt[clinical_data_gt['Class'] == 0]),
                         len(clinical_data_gt[clinical_data_gt['Class'] == 1])])

    return subclass_weight

def assign_combination_weights(subclass_labels, main_class_labels, combination_weights, subclass_to_combination_index):
    main_class_labels = torch.argmax(main_class_labels, dim=1)
    batch_combination_indices = [subclass_to_combination_index[(sub.item(), main.item())] for sub, main in zip(subclass_labels, main_class_labels)]
    batch_weights = combination_weights[batch_combination_indices]
    return torch.tensor(batch_weights, dtype=torch.float32)

def evaluate(all_outputs, all_labels, threshold):
    all_outputs = torch.cat(all_outputs, dim = 0)
    all_labels = torch.cat(all_labels, dim = 0).float().cpu().numpy()
    all_labels = all_labels[:, 1]

    predictions = torch.sigmoid(all_outputs)[:, 1].float().cpu().numpy()

    if threshold == 0:
        fpr, tpr, thresholds = metrics.roc_curve(all_labels, predictions)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
    else:
        best_thresh = threshold

    predictions_thresholded = (predictions > best_thresh)
    sensitivity = np.mean(predictions_thresholded[all_labels == 1])
    specificity = np.mean(predictions_thresholded[all_labels == 0] == False)

    fracture_roc_auc = metrics.roc_auc_score(y_true=all_labels, y_score= predictions, average=None)
    fracture_f1 = metrics.f1_score(y_true=all_labels, y_pred=predictions_thresholded, average='binary', pos_label=1)
    fracture_map = metrics.average_precision_score(y_true = all_labels, y_score = predictions)
    scores = {'AUC': fracture_roc_auc, 'F1': fracture_f1, 'mAP': fracture_map, 'Sensitivity': sensitivity, 'Specificity': specificity, 'Threshold': best_thresh}

    return scores