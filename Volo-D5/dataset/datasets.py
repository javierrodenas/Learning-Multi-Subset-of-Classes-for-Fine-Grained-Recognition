import os

import PIL.Image
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch.utils.data
import os
from torchvision.datasets.folder import pil_loader


class Food1k(torch.utils.data.Dataset):
    def __init__(self, path, transform, mode="test", dataset_reduction_factor=1):
        self.path = os.path.join(path, mode)
        self.mode = mode

        self.image_paths = []
        self.labels = []
        if mode == "test":
            self.image_paths = os.listdir(self.path)
        elif mode == "train" or mode == "val":
            label_dirs = os.listdir(self.path)
            for idx, label_dir in enumerate(label_dirs):
                img_paths = os.listdir(os.path.join(self.path, label_dir))[::dataset_reduction_factor]
                self.image_paths += img_paths
                self.labels += ([label_dir] * len(img_paths))

        self.len = len(self.image_paths)
        self.transform = transform

        self.max_size = None

    def __getitem__(self, item):
        img_path = self.image_paths[item]

        label = None
        if not self.mode == "test":
            label = self.labels[item]

        if self.mode == "test":
            full_img_path = os.path.join(self.path, img_path)
        else:
            full_img_path = os.path.join(self.path, label, img_path)

        raw_img = pil_loader(full_img_path)
        img_size = raw_img.size

        self.max_size = (max(img_size[0], self.max_size[0]), max(img_size[1], self.max_size[1])) if self.max_size is not None else img_size

        if self.transform is not None:
            img = self.transform(raw_img)
        else:
            img = raw_img

        return img, int(label) if label is not None else 0

    def __len__(self):
        return self.len


def read_single_column_txt(path):
    items = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                items += [line]
    return items


def fix_pil_image_size(img, max_size_wanted=512):
    """
    This method changes changes the image such that length of maximum dimension is 512
    """

    size = np.array(img.size)
    max_idx = np.argmax(size)
    max_size = size[max_idx]

    if max_size > max_size_wanted:
        scale_factor = max_size_wanted / max_size
        new_size = list(np.ceil(size * scale_factor))
        img.thumbnail(new_size, PIL.Image.ANTIALIAS)

    return img


class Food101(Dataset):
    def __init__(self, dataset_path, mode="train", transform=None,
                 return_id=False, dataset_reduction_factor=1):

        self.dataset_path = dataset_path
        self.annotation_path = dataset_path
        self.mode = mode
        self.transform = transform
        self.return_id = return_id


        classes = read_single_column_txt(os.path.join(self.annotation_path, "classes.txt"))
        self.num_classes = len(classes)
        self.img_paths = read_single_column_txt(os.path.join(self.annotation_path, "{}.txt".format(mode)))
        self.labels = read_single_column_txt(os.path.join(self.annotation_path, "labels.txt".format(mode)))

        labels_dict = dict()
        for index_label, label in enumerate(self.labels):
            label_modified = label.lower().replace(" ", '_')
            labels_dict[label_modified] = index_label

        self.img_labels = []
        for index_img, img_path in enumerate(self.img_paths):
            self.img_labels.append(labels_dict[img_path.split('/')[0]])
            self.img_paths[index_img] = 'images/' + img_path + '.jpg'


        assert len(self.img_paths) == len(self.img_labels)

        self.len = len(self.img_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_id = self.img_paths[item]
        img_path = os.path.join(self.dataset_path, img_id)
        img = pil_loader(img_path)
        img = fix_pil_image_size(img)

        if self.transform is not None:
            img = self.transform(img)

        img_label = int(self.img_labels[item])

        if not self.return_id:
            return img, img_label
        else:
            return img, img_label, img_id




class FoodX251Dataset(Dataset):
    def __init__(self, dataset_path, mode="train", transform=None,
                 return_id=False, dataset_reduction_factor=1):

        self.dataset_path = dataset_path
        self.annotation_path = dataset_path
        self.mode = mode
        self.transform = transform
        self.return_id = return_id


        classes = read_single_column_txt(os.path.join(self.annotation_path, "class_list.txt"))
        print(len(classes))
        self.num_classes = len(classes)

        dataframe_images = pd.read_csv(os.path.join(self.annotation_path, f"{mode}_labels.csv"))
        dataframe_images['img_name'] = mode + '_set/' + dataframe_images['img_name'].astype(str)

        self.img_paths = dataframe_images['img_name'].tolist()
        self.labels = dataframe_images['label'].tolist()

        assert len(self.img_paths) == len(self.labels)

        self.len = len(self.img_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_id = self.img_paths[item]
        img_path = os.path.join(self.dataset_path, img_id)
        img = pil_loader(img_path)
        img = fix_pil_image_size(img)

        if self.transform is not None:
            img = self.transform(img)

        img_label = int(self.labels[item])

        if not self.return_id:
            return img, img_label
        else:
            return img, img_label, img_id



def get_next_batch(loader, iterator):
    """
    Typically, a data loader will raise StopIteration exception when we reach the last batch, in case of hydra,
    that causes problems as we are iterating over multiple data loaders at the same time and they all will end at
    different epochs. This method allows us to bypass StopIteration and request as many batches as we want. In this
    setting we should manually determine how many iterations should happen at every "epoch".
    """
    if iterator is None:
        iterator = iter(loader)
    try:
        item = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        item = next(iterator)
    return item, iterator


if __name__ == '__main__':
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.Resize([64, 64])
        ]
    )
    dataset = LogMealTypesDataset("/mnt/8870443570442BEE/Data/LogMeal/FoodImageNet6.1/dataset_V6.1",
                                  "annotations/split_drinks",
                                  transform=transform)
    from torch.utils.data.dataloader import DataLoader
    from tqdm import tqdm

    loader = DataLoader(dataset, 128, True, num_workers=1)
    for i in tqdm(range(1000)):
        item, iterator = get_next_batch(loader, None)
