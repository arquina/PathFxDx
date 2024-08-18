import torch
from torch import nn, optim
from tqdm import tqdm
import os
import pandas as pd
from transformers import get_cosine_schedule_with_warmup
from utils import assign_combination_weights, evaluate

def train_class(model, data_loader, epoch_num, lr, weight_decay, save_dir, cuda, subclass_weight, Argument, cross_val = False):
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print("run in cpu")
    else:
        print("run in " + str(device))

    model.to(device)
    optimizer = optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=lr, weight_decay=weight_decay)

    # setup learning rate schedule and starting epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(epoch_num * len(data_loader['train']) / 40),
                                                num_training_steps=epoch_num * len(data_loader['train']))
    pos_weight = torch.tensor([1.0, 1.5]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    subclass_to_combination_index = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3, (2, 0): 4, (2, 1): 5, (3,0): 6, (3, 1): 7, (4, 0): 8, (4, 1): 9}

    train_loss_list = []
    test_loss_list = []
    val_loss_list = []
    external_test_loss_list = []

    best_train_auc = 0.5
    best_test_auc = 0.5
    best_val_auc = 0.5
    best_external_auc = 0.5

    best_train_map = 0.0
    best_test_map = 0.0
    best_val_map = 0.0
    best_external_map = 0.0

    train_auc_list = []
    test_auc_list = []
    val_auc_list = []
    external_test_auc_list = []

    train_map_list = []
    test_map_list = []
    val_map_list = []
    external_test_map_list = []

    train_threshold_list = []
    train_f1_list = []
    train_sensitivity_list = []
    train_specificity_list = []

    test_threshold_list = []
    test_f1_list = []
    test_sensitivity_list = []
    test_specificity_list = []

    external_f1_list = []
    external_sensitivity_list = []
    external_specificity_list = []

    val_threshold_list = []
    val_f1_list = []
    val_sensitivity_list = []
    val_specificity_list = []

    save = False

    if cross_val:
        phase_list = ['train', 'val', 'test', 'external_test']
    else:
        phase_list = ['train', 'test', 'external_test']

    for epoch in range(epoch_num):
        for mode in phase_list:
            if mode == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            total = 0
            outputs_list = []
            labels_list = []

            with tqdm(total=len(data_loader[mode])) as pbar:
                for i, data in enumerate(data_loader[mode]):
                    if Argument.dataset == 'Dual':
                        original_inputs, fracture_inputs, target, location = data
                        original_inputs = original_inputs.to(device)
                        fracture_inputs = fracture_inputs.to(device)
                    else:
                        inputs, target, location = data
                        inputs = inputs.to(device)
                    labels = target.to(device).float()
                    locations = location.to(device)
                    weight = assign_combination_weights(locations, labels, subclass_weight, subclass_to_combination_index)
                    optimizer.zero_grad()
                    if Argument.dataset == 'Dual':
                        outputs = model(original_inputs, fracture_inputs)
                    else:
                        outputs = model(inputs)

                    total += labels.size(0)
                    adjusted_weights = weight.unsqueeze(1).repeat(1, outputs.shape[1]).to(device)
                    loss = criterion(outputs, labels)
                    loss = adjusted_weights * loss
                    loss = loss.mean()

                    outputs_list.append(outputs.detach())
                    labels_list.append(labels.detach())

                    if mode == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm = 1.0,
                                                       error_if_nonfinite=True)
                        optimizer.step()
                        scheduler.step()

                    pbar.set_description("Loss: %0.5f, lr: %0.8f " % (loss, optimizer.param_groups[0]['lr']))
                    running_loss += loss.item()
                    pbar.update()

            epoch_loss = running_loss / len(data_loader[mode])
            if mode == "train":
                print(epoch)
                train_loss_list.append(epoch_loss)
                scores = evaluate(outputs_list, labels_list, threshold = 0)
                print("train_eval")
                print("train_loss: ", epoch_loss,
                          ' train_auc: ', scores['AUC'],
                          ' train_mAP: ', scores['mAP'],
                          ' train_threshold: ', scores['Threshold'],
                          ' train_f1: ', scores['F1'],
                          ' train_sensitivity: ', scores['Sensitivity'],
                          ' train_specificity: ', scores['Specificity'])

                train_auc_list.append(scores['AUC'])
                train_map_list.append(scores['mAP'])
                train_threshold_list.append(scores['Threshold'])
                train_f1_list.append(scores['F1'])
                train_sensitivity_list.append(scores['Sensitivity'])
                train_specificity_list.append(scores['Specificity'])

                if best_train_auc < scores['AUC']:
                    best_train_auc = scores['AUC']
                if best_train_map < scores['mAP']:
                    best_train_map = scores['mAP']
            if mode == "val":
                print(epoch)
                print('val')
                scores = evaluate(outputs_list, labels_list, threshold=0)
                epoch_threshold = scores['Threshold']
                val_loss_list.append(epoch_loss)
                print("val_loss: ", epoch_loss,
                      ' val_auc: ', scores['AUC'],
                      ' val_mAP: ', scores['mAP'],
                      ' val_threshold: ', scores['Threshold'],
                      ' val_f1: ', scores['F1'],
                      ' val_sensitivity: ', scores['Sensitivity'],
                      ' val_specificity: ', scores['Specificity'])
                val_auc_list.append(scores['AUC'])
                val_map_list.append(scores['mAP'])
                val_threshold_list.append(scores['Threshold'])
                val_f1_list.append(scores['F1'])
                val_sensitivity_list.append(scores['Sensitivity'])
                val_specificity_list.append(scores['Specificity'])

                if best_val_auc < scores['AUC']:
                    best_val_auc = scores['AUC']
                    if Argument.save:
                        if save == False:
                            save = True
                            model_save_dir = os.path.join(save_dir, 'model')
                            if os.path.exists(model_save_dir) is False:
                                os.mkdir(model_save_dir)
                            torch.save(model.state_dict(), os.path.join(model_save_dir, 'valAUC_' + str(scores['AUC']) + '_' + 'valmAP_' + str(scores['mAP']) + '_' +  str(epoch) + '.pt'))
                if best_val_map < scores['mAP']:
                    best_val_map = scores['mAP']
                    if Argument.save:
                        if save == False:
                            save = True
                            model_save_dir = os.path.join(save_dir, 'model')
                            if os.path.exists(model_save_dir) is False:
                                os.mkdir(model_save_dir)
                            torch.save(model.state_dict(), os.path.join(model_save_dir, 'valAUC_' + str(scores['AUC']) + '_' + 'valmAP_' + str(scores['mAP']) + '_' +  str(epoch) + '.pt'))

            if mode == "test":
                print(epoch)
                test_loss_list.append(epoch_loss)
                if cross_val:
                    scores = evaluate(outputs_list, labels_list, threshold= epoch_threshold)
                else:
                    scores = evaluate(outputs_list, labels_list, threshold = 0)
                    epoch_threshold = scores['Threshold']
                print('test')
                print("test_loss: ", epoch_loss,
                          ' test_auc: ', scores['AUC'],
                          ' test_mAP: ', scores['mAP'],
                          ' test_threshold: ', scores['Threshold'],
                          ' test_f1: ', scores['F1'],
                          ' test_sensitivity: ', scores['Sensitivity'],
                          ' test_specificity: ', scores['Specificity'])

                test_auc_list.append(scores['AUC'])
                test_map_list.append(scores['mAP'])
                test_f1_list.append(scores['F1'])
                test_threshold_list.append(scores['Threshold'])
                test_sensitivity_list.append(scores['Sensitivity'])
                test_specificity_list.append(scores['Specificity'])
                if best_test_auc < scores['AUC']:
                    best_test_auc = scores['AUC']
                    if Argument.save:
                        if save == False:
                            save = True
                            model_save_dir = os.path.join(save_dir, 'model')
                            if os.path.exists(model_save_dir) is False:
                                os.mkdir(model_save_dir)
                            torch.save(model.state_dict(), os.path.join(model_save_dir, 'testAUC_' + str(scores['AUC']) + '_' + 'testmAP_' + str(scores['mAP']) + '_' +  str(epoch) + '.pt'))
                if best_test_map < scores['mAP']:
                    best_test_map = scores['mAP']
                    if Argument.save:
                        if save == False:
                            save = True
                            model_save_dir = os.path.join(save_dir, 'model')
                            if os.path.exists(model_save_dir) is False:
                                os.mkdir(model_save_dir)
                            torch.save(model.state_dict(), os.path.join(model_save_dir, 'test_AUC_' + str(scores['AUC']) + '_' + 'test_mAP_' + str(scores['mAP']) + '_' + str(epoch)  + '.pt'))

            if mode == "external_test":
                print(epoch)
                print('external_test')
                scores = evaluate(outputs_list, labels_list, threshold=epoch_threshold)
                external_test_loss_list.append(epoch_loss)
                print("external_loss: ", epoch_loss,
                      ' external_auc: ', scores['AUC'],
                      ' external_mAP: ', scores['mAP'],
                      ' external_threshold: ', scores['Threshold'],
                      ' external_f1: ', scores['F1'],
                      ' external_sensitivity: ', scores['Sensitivity'],
                      ' external_specificity: ', scores['Specificity'])
                external_test_auc_list.append(scores['AUC'])
                external_test_map_list.append(scores['mAP'])
                external_f1_list.append(scores['F1'])
                external_sensitivity_list.append(scores['Sensitivity'])
                external_specificity_list.append(scores['Specificity'])

                if best_external_auc <= scores['AUC']:
                    best_external_auc = scores['AUC']
                    if Argument.save:
                        if save == False:
                            save = True
                            model_save_dir = os.path.join(save_dir, 'model')
                            if os.path.exists(model_save_dir) is False:
                                os.mkdir(model_save_dir)
                            torch.save(model.state_dict(), os.path.join(model_save_dir, 'external_AUC_' + str(scores['AUC']) + '_external_mAP_' + str(scores['mAP']) + '_' + str(epoch) + '.pt'))
                if best_external_map <= scores['mAP']:
                    best_external_map = scores['mAP']
                    if Argument.save:
                        if save == False:
                            save = True
                            model_save_dir = os.path.join(save_dir, 'model')
                            if os.path.exists(model_save_dir) is False:
                                os.mkdir(model_save_dir)
                            torch.save(model.state_dict(), os.path.join(model_save_dir, 'external_AUC_' + str(scores['AUC']) + '_externalt_mAP_' + str(scores['mAP']) + '_' + str(epoch) + '.pt'))

                save = False

    print('Best test AUC: ', best_test_auc)
    print('Best test mAP: ', best_test_map)

    print('Best external test mAP: ', best_external_map)
    print('Best external test AUC: ', best_external_auc)

    if cross_val:
        df = pd.DataFrame(list(zip(train_auc_list, val_auc_list, test_auc_list, external_test_auc_list,
                                   train_map_list, val_map_list, test_map_list, external_test_map_list,
                                   train_f1_list, val_f1_list, test_f1_list, external_f1_list,
                                   train_threshold_list, val_threshold_list, test_threshold_list,
                                   train_sensitivity_list, val_sensitivity_list, test_sensitivity_list,
                                   external_sensitivity_list,
                                   train_specificity_list, val_specificity_list, test_specificity_list,
                                   external_specificity_list,
                                   list(range(len(train_auc_list))))),
                          columns=['train_AUC', 'val_AUC', 'test_AUC', 'external_test_AUC',
                                   'train_mAP', 'val_mAP', 'test_mAP', 'external_test_mAP',
                                   'train_F1', 'val_F1', 'test_F1', 'external_F1',
                                   'train_threshold', 'val_threshold', 'test_threshold',
                                   'train_sensitivity', 'val_sensitivity', 'test_sensitivity', 'external_sensitivity',
                                   'train_specificity', 'val_specificity', 'test_specificity', 'external_specificity',
                                   'Epoch'])
        if Argument.save:
            df.to_csv(os.path.join(save_dir, 'result.csv'), index=False)
        return best_val_auc, best_val_map
    else:
        df = pd.DataFrame(list(zip(train_auc_list, test_auc_list, external_test_auc_list,
                                   train_map_list, test_map_list, external_test_map_list,
                                   train_f1_list, test_f1_list, external_f1_list,
                                   train_threshold_list, test_threshold_list,
                                   train_sensitivity_list, test_sensitivity_list, external_sensitivity_list,
                                   train_specificity_list, test_specificity_list, external_specificity_list,
                                   list(range(len(train_auc_list))))),
                          columns=['train_AUC', 'test_AUC', 'external_test_AUC',
                                   'train_mAP', 'test_mAP', 'external_test_mAP',
                                   'train_F1',  'test_F1', 'external_F1',
                                   'train_threshold', 'test_threshold',
                                   'train_sensitivity', 'test_sensitivity', 'external_sensitivity',
                                   'train_specificity', 'test_specificity', 'external_specificity',
                                   'Epoch'])
        if Argument.save:
            df.to_csv(os.path.join(save_dir, 'result.csv'), index=False)