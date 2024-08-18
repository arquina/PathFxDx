Deep Learning Model for Differentiating Between Neoplastic Pathologic Fracture and Nonpathologic Fracture from Plain Hip Radiographs
=====================

## Dependencies
* To install the dependencies for this project, see the "requirements.yaml"
* Tested on Nvidia RTX6000

## Prerequisite
* **Data_root_dir**: Must contain original/fracture directory which has x-ray image the file name should be same (for attention analysis you need json file.
* **Clinical_data**: xlsx file with column Image_Name(file_name), Hospital, Internal_test_set(for train/test split - train: 0, test: 1), Class(Nonpathologic:0, Neoplastic pathologic fracture: 1)

## Trained model
* Trained model will be provided by the Google drive. (Dual/ Fracture/ Original)
* Google drive address: <https://drive.google.com/drive/folders/15Re_Zbwf38NGaro1WFldc8rPy25Opffw?usp=drive_link>

## Step 1: 5 Fold Cross validation
* You can perform 5-fold cross validation using the below code (Example)
'''
python main.py --save_dir /home/seob/PathFxDx/result --rootdir <path to data_root_dir> --clinical_data <path to clinical data> --batch_size 32 --epoch 100 --lr 1e-3 --weight_decay 1e-3 --cuda cuda:0 --FF_number 5 --dataset Dual  --cross_val --save
'''

BiNEL (http://binel.snu.ac.kr) - This code is made available under the MIT License and is available for non-commercial academic purposes
