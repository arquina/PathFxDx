from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import json

def get_mean_std(dataset):
    if dataset == 'original':
        mean = 0.4295
        std = 0.2342

    elif dataset == 'fracture':
        mean = 0.6194
        std = 0.1335

    return mean, std

def add_kfold_to_df(df, n_fold, seed):
    df['stratify_col'] = df['Class'].astype(str) + '_' + df['Location']
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for fold, (_, val_) in enumerate(skf.split(X=df, y=df.stratify_col)):
        df.loc[df.index[val_], "kfold"] = int(fold)

    df['kfold'] = df['kfold'].astype(int)
    kfold = df['kfold'].values

    return kfold

class xray_dataset_class(Dataset):
    def __init__(self, df, phase, image_dir):
        self.root_dir = image_dir
        self.dataset = image_dir.split('/')[-1]
        self.df = df
        self.filelist = self.df['Image_Name'].tolist()
        self.locationlist = self.df['Location'].tolist()
        self.location_dict = {'Subtrochanteric area': 0,
                              'Intertrochanteric area': 1,
                              'Neck': 2,
                              'Isolated LT': 3,
                              'Isolated GT': 4}

        mean_value, std_value = get_mean_std(self.dataset)
        if phase == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4),
                transforms.RandomRotation(30),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[mean_value] * 3,
                                     std=[std_value] * 3)])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[mean_value] * 3,
                                     std=[std_value] * 3)
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.filelist[idx]
        img_path = os.path.join(self.root_dir, image_name)
        original_image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(original_image)
        target_df = self.df[self.df['Image_Name'] == image_name]
        location = target_df['Location'].item()
        location_label = self.location_dict[location]
        target = target_df['Class'].item()
        target = [[1, 0], [0, 1]][int(target)]
        target = torch.as_tensor(target)
        return image, target, location_label

class xray_dataset_class_dual(Dataset):
    def __init__(self, df, phase, image_dir, bounding_box = None):
        self.root_dir = image_dir
        self.original_root_dir = os.path.join(self.root_dir, 'original')
        self.fracture_root_dir = os.path.join(self.root_dir, 'fracture')
        self.df = df
        self.filelist = self.df['Image_Name'].tolist()
        self.locationlist = self.df['Location'].tolist()
        self.location_dict = {'Subtrochanteric area': 0,
                              'Intertrochanteric area': 1,
                              'Neck': 2,
                              'Isolated LT': 3,
                              'Isolated GT': 4}
        self.bounding_box = bounding_box
        self.json_dir = "/home/seob/class_project/20231106_pathologic/"


        original_mean, original_std = get_mean_std('original')
        fracture_mean, fracture_std = get_mean_std('fracture')

        if phase == 'train':
            self.transform_original = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4),
                transforms.RandomRotation(30),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[original_mean] * 3,
                                     std=[original_std] * 3)])

            self.transform_fracture = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4),
                transforms.RandomRotation(30),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[fracture_mean] * 3,
                                     std=[fracture_std] * 3)])
        else:
            self.transform_original = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[original_mean] * 3,
                                     std=[original_std] * 3)
            ])
            self.transform_fracture = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[fracture_mean] * 3,
                                     std=[fracture_std] * 3)
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        size = self.bounding_box
        image_name = self.filelist[idx]
        sample = image_name.split('.')[0]
        original_img_path = os.path.join(self.original_root_dir, image_name)
        fracture_img_path = os.path.join(self.fracture_root_dir, image_name)
        if size != None:
            json_file = os.path.join(self.json_dir, sample + '.json')
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            fracture_data = json_data['shapes'][1]
            if fracture_data['label'] != 'Fracture':
                fracture_data = json_data['shapes'][0]
            selected_bbox = fracture_data['points']
            original_image_original = Image.open(original_img_path).convert('RGB')
            center_x = (selected_bbox[0][0] + selected_bbox[1][0]) // 2
            center_y = (selected_bbox[0][1] + selected_bbox[1][1]) // 2

            # Define the square crop dimensions
            half_size = size // 2
            left = max(center_x - half_size, 0)
            top = max(center_y - half_size, 0)
            right = left + size
            bottom = top + size

            # Adjust the crop area if it goes out of image boundaries
            if right > original_image_original.width:
                right = original_image_original.width
                left = max(right - size, 0)
            if bottom > original_image_original.height:
                bottom = original_image_original.height
                top = max(bottom - size, 0)

            # Crop the image
            fracture_img_original = original_image_original.crop((left, top, right, bottom))
        else:
            original_image_original = Image.open(original_img_path).convert('RGB')
            fracture_img_original = Image.open(fracture_img_path).convert('RGB')
            if self.transform_fracture:
                original_image = self.transform_original(original_image_original)
                fracture_image = self.transform_fracture(fracture_img_original)

        if self.transform_fracture:
            original_image = self.transform_original(original_image_original)
            fracture_image = self.transform_fracture(fracture_img_original)
        target_df = self.df[self.df['Image_Name'] == image_name]
        location = target_df['Location'].item()
        location_label = self.location_dict[location]
        target = target_df['Class'].item()
        target = [[1, 0], [0, 1]][int(target)]
        target = torch.as_tensor(target)
        return original_image, fracture_image, target, location_label

class xray_dataset_class_val(Dataset):
    def __init__(self, df, image_dir):
        self.root_dir = image_dir
        self.original_root_dir = os.path.join(self.root_dir, 'original')
        self.fracture_root_dir = os.path.join(self.root_dir, 'fracture')
        self.df = df
        self.filelist = self.df['Image_Name'].tolist()
        self.locationlist = self.df['Location'].tolist()

        original_mean, original_std = get_mean_std('original')
        fracture_mean, fracture_std = get_mean_std('fracture')

        self.transform_original = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[original_mean] * 3,
                                 std=[original_std] * 3)
        ])
        self.transform_fracture = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[fracture_mean] * 3,
                                 std=[fracture_std] * 3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.filelist[idx]
        original_img_path = os.path.join(self.original_root_dir, image_name)
        fracture_img_path = os.path.join(self.fracture_root_dir, image_name)
        original_image_original = Image.open(original_img_path).convert('RGB')
        fracture_image_original = Image.open(fracture_img_path).convert('RGB')
        original_image = self.transform_original(original_image_original)
        fracture_image = self.transform_fracture(fracture_image_original)
        target_df = self.df[self.df['Image_Name'] == image_name]
        target = target_df['Class'].item()
        target = [[1, 0], [0, 1]][int(target)]
        target = torch.as_tensor(target)
        return original_image, fracture_image, target, original_image_original, fracture_image_original, image_name