from torchvision.datasets import ImageFolder
import numpy as np

class AlbumentationsDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.albu_transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = np.array(self.loader(path))

        if self.albu_transform:
            image = self.albu_transform(image=image)["image"]

        return image, label
