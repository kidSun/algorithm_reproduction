from PIL import Image
import torch.utils.data as data
import torch
import os
class MyDataset(data.Dataset):
    def __init__(
            self
            ,root
            ,class_map_path
            ,load_bytes=False
            ,transform=None):
        self.root = root
        self.load_bytes = load_bytes
        samples = []
        with open(class_map_path) as f:
            for s in f:
                k = s.split()[0]
                k = os.path.join(root,k)
                v = int(s.split()[1])
                samples.append((k,v))
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)           

