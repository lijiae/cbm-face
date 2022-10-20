from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision
from PIL import Image
import os

class img_attr_label(Dataset):
    def __init__(self,imagepath,csvfile,idfile):
        self.mean_bgr=np.array([91.4953, 103.8827, 131.0912])
        self.maadfile=csvfile
        self.idfile=idfile
        self.dir=imagepath

    def __len__(self):
        return len(self.idfile)

    def __getitem__(self, index):
        # Sample
        attr_row=self.maadfile.iloc[index]
        id=self.idfile.iloc[index]
        assert id.name==attr_row.name

        # data and label information
        imgname=attr_row.name
        label=torch.tensor(int(id)).long()
        attr=torch.tensor(np.array(attr_row)).float()
        data = torchvision.transforms.Resize(224)(Image.open(os.path.join(self.dir,imgname)))
        data = np.array(data, dtype=np.uint8)
        data = self.transform(data)
        # label = np.int32(label)

        return data.float(), attr,label

    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img

