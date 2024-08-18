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

```
python main.py --rootdir <path to data_root_dir> --clinical_data <path to clinical data> --batch_size 32 --epoch 100 --lr 1e-3 --weight_decay 1e-3 --cuda cuda:0 --FF_number 5 --dataset Dual --train_hospital <Name of train data's hospital> --test_hospital <Name of test data's hospital> --cross_val --save --save_dir <path you want to save the result>
```

## Step 2: Train model
* You can train your own model using train/ test/ external_test data using the below code (Example)

```
python main.py --save_dir /home/seob/PathFxDx/result --rootdir <path to data_root_dir> --clinical_data <path to clinical data> --batch_size 32 --epoch 100 --lr 1e-3 --weight_decay 1e-3 --cuda cuda:0 --dataset Dual --train_hospital <Name of train data's hospital> --test_hospital <Name of test data's hospital> --save --save_dir <path you want to save the result>
```

## Step 3: Analyze the 5 fold cross validation result & Final model result
* After performing 5 fold cross validation you can analyze the result. The main purpose of this code is comparing different dataset type. (Dual/ Fracture/ Original). Script is in Interpretation_utils.
* target_metric is the target_metric to select the best model/ probability_calculation: Need only once at the first code run. 
* Example code
```
python analyze_five_fold.py --rootdir <path to data_root_dir> --clinical_data <path to clinical data> --script_dir <root path of the PathFxDx which contains Dataloader folder> --cuda cuda:0 --FF_number 5 --analysis_dir <path to the directory name '5fold_comparison'> --Original_analysis_dir <path to original model root dir> --Fracture_analysis_dir <path to fracture model root dir> --Dual_analysis_dir <path to dual model root dir> --target_metric AUPRC --probability_calculation
```

* Also you can run analyze.py to interpret the final trained model after model training
```
python analyze.py --rootdir <path to data_root_dir> --clinical_data <path to clinical data> --script_dir <root path of the PathFxDx which contains Dataloader folder> --cuda cuda:0 --dataset Dual --analysis_dir <path to the save_dir> --model_dir <path to model root dir> --model_name <model_file name> 
```
* If you want to compare your result with expert, you need to have expert data which has column name 'Expert'

## Step 4: Attention analysis
* You can run attention analysis in original model and fracture model
* Example code
```
python attention_map_interpretation.py --rootdir <path to data_root_dir> --clinical_data <path to clinical data> --script_dir <root path of the PathFxDx which contains Dataloader folder> --cuda cuda:0 --dataset Original --analysis_dir <path to the save_dir> --model_dir <path to model root dir> --model_name <model_file name> 
```

## Step 5: GUI
* If you only want to run the model using the GUI, run GUI.py after downloading the trained model in the google drive.


BiNEL (http://binel.snu.ac.kr) - This code is made available under the MIT License and is available for non-commercial academic purposes
