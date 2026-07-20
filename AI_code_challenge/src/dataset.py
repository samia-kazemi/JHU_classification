import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from AI_code_challenge.src.preprocessing import resize_image, apply_whitebalancing, add_augmentations
from AI_code_challenge.src.utils import add_data_split

class MosDataset(Dataset):
    def __init__(self, config, split = "Train"):
        self.data_dir = config["data_dir"]
        self.data_subfolders = config['data_subfolders']
        self.split = split
        if self.split == 'Test':
            self.mode = "Test"
        else:
            self.mode = config['mode']
        self.img_size = config["image_size"]
        self.wb = config["whitebalance"]
        self.exp_folder = Path('experiments') / config['exp_name']
        if self.mode == 'Train' and self.split == 'Train':
            # Initial processing of the datasheets and save copy to experiment folder
            self.create_combined_datasheet()  # Create combined datasheet, add split and label columns
        else:
            self.images_df = pd.read_csv(self.exp_folder / 'existing_data.csv', dtype={"y": "int64",
                                                                                       "ImageFolder": "str"})
        self.prep_datasheet()
        #print(self.__len__())
        #self.__getitem__(10)

    def prep_datasheet(self):
        # Only retain specimens for the current split
        self.images_df = self.images_df[self.images_df['Split'] == self.split]
        if self.mode == 'Test':
            self.species_names = []
            for y in range(max(set(self.images_df['y'])) + 1):
                sub_df = self.images_df[self.images_df['y'] == y]
                self.species_names.append(list(set(sub_df['SpeciesLabel']))[0])
            print("self.species_names = ", self.species_names)

    def __getitem__(self, ind):
        row = self.images_df.iloc[ind]
        image_fullpath = Path(self.data_dir) / row['ImageFolder'] / 'downloaded_images' / row['DownloadedFilename']
        img = Image.open(image_fullpath)
        img_array = np.asarray(img)

        # Crop out top portion of the image
        img_array = img_array[round(img_array.shape[0] / 3):]

        # Resize image
        img_array = resize_image(Image.fromarray(img_array), self.img_size)

        # Apply white balancing to the image
        if self.wb:
            img_wb = apply_whitebalancing(img_array)
        ''' img_array = Image.fromarray(img_array)
        img_wb = Image.fromarray(img_wb)
        img_array.show()
        img_wb.show() '''

        # Add augmentations in Train model
        if self.mode == "Train":
            img_wb = add_augmentations(img_wb)

        # Convert to channel first
        img_wb = np.transpose(img_wb, [2, 0, 1])
    
        return img_wb.astype(np.float32), row['y']

    def __len__(self):
        return len(self.images_df)

    def create_combined_datasheet(self):
        # Create combined csv datasheet, only include rows for image files that exists
        self.images_df = pd.DataFrame()
        for data_subfolder in self.data_subfolders:
            print(data_subfolder)
            data_dir = Path(self.data_dir) / str(data_subfolder)
            df = pd.read_csv(data_dir / 'specimens_modeling_master.csv')
            df['ImageFolder'] = data_subfolder
            exists = [-1] * len(df)
            for idx, row in df.iterrows():
                if type(row['DownloadedFilename']) == str:
                    image_fullpath = data_dir / 'downloaded_images' / row['DownloadedFilename']
                    if Path.exists(image_fullpath):
                        exists[idx] = 1
            df['ImagefileAvailable'] = exists
            print("before: ", len(df))
            df = df[df['ImagefileAvailable'] == 1]
            print("after: ", len(df))
            self.images_df = pd.concat((self.images_df, df), axis=0)

        # Add y (label) column
        sp_to_y = {x: i for i, x in enumerate(sorted(set(df['SpeciesLabel'])))}
        self.images_df['y'] = self.images_df['SpeciesLabel'].map(sp_to_y)
        self.images_df = self.images_df.dropna(subset=['y'])
        self.images_df['y'] = self.images_df['y'].astype(int)

        # Add split column
        self.images_df = add_data_split(self.images_df, ratio = [0.7, 0.15, 0.15])

        # Save combined datasheet to experiment folder
        self.images_df.to_csv(self.exp_folder / 'existing_data.csv', index=False)
