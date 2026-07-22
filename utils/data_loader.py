# data_loader.py

import os
from torch.utils.data import DataLoader # type: ignore
from torchvision import transforms # type: ignore
from pycocotools.coco import COCO
from PIL import Image
import torch # type: ignore

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, anns

    def __len__(self):
        return len(self.ids)


def get_loader(image_root, ann_path, batch_size=8, shuffle=True, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = COCODataset(root=image_root, annFile=ann_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
