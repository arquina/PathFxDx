import argparse
from torch.utils.data import DataLoader
from time import strftime, localtime, time
import os
from Dataloader import xray_dataset_class, xray_dataset_class_dual
from Dataloader import add_kfold_to_df
from model import  CustomViT
import timm
import pandas as pd
from utils import get_class_balanced_weight
from train import train_class

def Parser_main():
    parser = argparse.ArgumentParser(description="Bone fracture classification")
    parser.add_argument("--rootdir" , help="dataset rootdir", type = str)
    parser.add_argument("--batch_size", default= 32 , help="train, test batch_size", type=int)
    parser.add_argument("--epoch", default= 1, help="epoch number", type=int)
    parser.add_argument("--lr", default = 1e-3, help = "Learning rate", type = float)
    parser.add_argument("--weight_decay", default = 1e-3, help = "Weight decay", type = float)
    parser.add_argument("--cuda", default = 'cuda:0', type = str, help = 'cuda device or cpu')
    parser.add_argument("--random_seed", default = 1234567, type = int)
    parser.add_argument("--save_dir", default = "./result/", help = "result save directory", type = str)
    parser.add_argument("--cross_val", action = 'store_true', default = False)
    parser.add_argument("--FF_number", default = 5, type = int)
    parser.add_argument('--dataset', default = 'Dual', help = 'original, fracture, Dual')
    parser.add_argument('--clinical_data', help = 'clinical_data')
    parser.add_argument('--save', default = False, action = 'store_true', help = 'save the result')
    parser.add_argument('--bbox_size', default = None, type = int, help = 'if you want to check the various size of square bbox use this parameter')
    parser.add_argument('--train_hospital', default = 'SNUH', type = str, help = 'Train dataset hospital')
    parser.add_argument('--test_hospital', default = 'SNUBH', type = str, help = 'External test dataset hospital')
    return parser.parse_args()

def main():
    tm = localtime(time())
    cur_time = strftime('%Y-%m-%d_%H:%M:%S', tm)
    Argument = Parser_main()
    cur_time += '_epoch_' + str(Argument.epoch) + '_lr_' + str(Argument.lr) + '_decay_' + str(Argument.weight_decay) + '_box_size_' + str(Argument.bbox_size)
    if Argument.save:
        if os.path.exists(Argument.save_dir) is False:
            os.mkdir(Argument.save_dir)
        if Argument.cross_val:
            result_dir = os.path.join(Argument.save_dir, '5fold_comparison')
            if os.path.exists(result_dir) is False:
                os.mkdir(result_dir)
            if Argument.bbox_size != None:
                result_dir = os.path.join(result_dir, str(Argument.bbox_size))
                if os.path.exists(result_dir) is False:
                    os.mkdir(result_dir)
        else:
            result_dir = Argument.save_dir

        result_dir = os.path.join(result_dir, Argument.dataset)
        if os.path.exists(result_dir) is False:
            os.mkdir(result_dir)

        result_dir = os.path.join(result_dir, 'All')
        if os.path.exists(result_dir) is False:
            os.mkdir(result_dir)

        result_dir = os.path.join(result_dir, cur_time)
        if os.path.exists(result_dir) is False:
            os.mkdir(result_dir)

    # K-fold cross validation
    if Argument.cross_val:
        clinical_data = pd.read_excel(Argument.clinical_data)

        # Set train data except test data and external test data and split the data with 5 fold preserving the original distribution
        train_data = clinical_data[clinical_data['Hospital'].isin([Argument.train_hospital])].copy()
        train_data = train_data[train_data['Internal_test_set'] == 0]
        _ = add_kfold_to_df(train_data, Argument.FF_number, Argument.random_seed)
        subclass_weight = get_class_balanced_weight(train_data)

        Fi_number_list = []
        Fi_result_list = []
        Fi_mAP_list = []

        for Fi in range(Argument.FF_number):
            Fi_number_list.append(Fi)
            if Argument.save:
                if os.path.exists(os.path.join(result_dir, str(Fi))) is False:
                    os.mkdir(os.path.join(result_dir, str(Fi)))

            TrainFF_set = train_data[train_data['kfold'] != int(Fi)]
            ValFF_set = train_data[train_data['kfold'] == int(Fi)]
            test_df = clinical_data[clinical_data['Internal_test_set'] == 1]
            external_test_df = clinical_data[clinical_data['Hospital'].isin([Argument.test_hospital])]

            if Argument.save:
                TrainFF_set.to_csv(os.path.join(result_dir, str(Fi), 'patient_train_set.csv'), index=False)
                ValFF_set.to_csv(os.path.join(result_dir, str(Fi), 'patient_val_set.csv'), index=False)

            print("Load classification dataset")
            if Argument.dataset != 'Dual':
                train_dataset = xray_dataset_class(TrainFF_set, 'train', os.path.join(Argument.rootdir, Argument.dataset))
                val_dataset = xray_dataset_class(ValFF_set, 'test', os.path.join(Argument.rootdir, Argument.dataset))
                test_dataset = xray_dataset_class(test_df, 'test', os.path.join(Argument.rootdir, Argument.dataset))
                external_test_dataset = xray_dataset_class(external_test_df, 'test', os.path.join(Argument.rootdir, Argument.dataset))
            else:
                train_dataset = xray_dataset_class_dual(TrainFF_set, 'train', Argument.rootdir, bounding_box = Argument.bbox_size)
                val_dataset = xray_dataset_class_dual(ValFF_set, 'test', Argument.rootdir, bounding_box = Argument.bbox_size)
                test_dataset = xray_dataset_class_dual(test_df, 'test', Argument.rootdir, bounding_box = Argument.bbox_size)
                external_test_dataset = xray_dataset_class_dual(external_test_df, 'test', Argument.rootdir, bounding_box = Argument.bbox_size)

            print("Dataset Loading Complete")
            print("Load Dataloader")
            train_dataloader = DataLoader(train_dataset, batch_size=Argument.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=Argument.batch_size, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=Argument.batch_size, shuffle=False)
            external_test_dataloader = DataLoader(external_test_dataset, batch_size=Argument.batch_size, shuffle=False)
            print("Dataloader Loading Complete")

            dataloader = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader, 'external_test': external_test_dataloader}

            print("Load classification model")
            if Argument.dataset == 'Dual':
                model = CustomViT()
            else:
                model = timm.create_model('vit_base_patch16_224_dino', pretrained=True, num_classes=2)
            print('Model call complete')

            print('Train start!')
            Fi_auc, Fi_mAP = train_class(model, dataloader, Argument.epoch, Argument.lr, Argument.weight_decay, os.path.join(result_dir, str(Fi)), Argument.cuda, subclass_weight, Argument, cross_val = True)
            print('Train Finish!')
            Fi_result_list.append(Fi_auc)
            Fi_mAP_list.append(Fi_mAP)

        if Argument.save:
            df = pd.DataFrame(list(zip(Fi_number_list, Fi_result_list, Fi_mAP_list)), columns = ['Fold number', 'Best_AUC', 'Best_mAP'])
            df.to_csv(os.path.join(result_dir, 'Cross_Validation_result.csv'), index = False)

    else:
        print("Load classification dataset")
        clinical_data = pd.read_excel(Argument.clinical_data)

        train_data = clinical_data[clinical_data['Hospital'].isin([Argument.train_hospital])].copy()
        subclass_weight = get_class_balanced_weight(train_data)
        external_test_df = clinical_data[clinical_data['Hospital'].isin([Argument.test_hospital])].copy()

        train_df = train_data[train_data['Internal_test_set'] == 0]
        test_df = train_data[train_data['Internal_test_set'] == 1]

        if Argument.dataset != 'Dual':
            train_dataset = xray_dataset_class(train_df, 'train', os.path.join(Argument.rootdir, Argument.dataset))
            test_dataset = xray_dataset_class(test_df, 'test', os.path.join(Argument.rootdir, Argument.dataset))
            external_test_dataset = xray_dataset_class(external_test_df, 'test', os.path.join(Argument.rootdir, Argument.dataset))
        else:
            train_dataset = xray_dataset_class_dual(train_df, 'train', Argument.rootdir, bounding_box = Argument.bbox_size)
            test_dataset = xray_dataset_class_dual(test_df, 'test', Argument.rootdir,  bounding_box = Argument.bbox_size)
            external_test_dataset = xray_dataset_class_dual(external_test_df, 'test', Argument.rootdir, bounding_box = Argument.bbox_size)

        print("Load Dataloader")
        train_dataloader = DataLoader(train_dataset, batch_size = Argument.batch_size, shuffle = True)
        test_dataloader = DataLoader(test_dataset, batch_size = Argument.batch_size, shuffle = False)
        external_test_dataloader = DataLoader(external_test_dataset, batch_size = Argument.batch_size, shuffle = False)
        dataloader = {'train': train_dataloader, 'test': test_dataloader, 'external_test': external_test_dataloader}
        print("Data loader loading complete")

        print("Load classification model")
        if Argument.dataset == 'Dual':
            model = CustomViT()
        else:
            model = timm.create_model('vit_base_patch16_224_dino', pretrained=True, num_classes=2)
        print('Model call complete')

        print('Train start!')
        train_class(model, dataloader, Argument.epoch, Argument.lr, Argument.weight_decay, result_dir, Argument.cuda, subclass_weight, Argument)
        print('Train Finish!')

if __name__ == "__main__":
    main()