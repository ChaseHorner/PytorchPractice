import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, base_path, is_train=True):
        """
        base_path: path to 'train' or 'val' folder
        """
        self.is_train = is_train
        self.low_res_path = os.path.join(base_path, 'low_res')
        self.high_res_path = os.path.join(base_path, 'high_res')

        self.low_res_images = sorted(os.listdir(self.low_res_path))
        self.high_res_images = sorted(os.listdir(self.high_res_path))

        assert len(self.low_res_images) == len(self.high_res_images), \
            "Number of low-res and high-res images should match"

        # Transformation: convert to tensor (you can add more if needed)
        self.transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        # Load images
        low_img = Image.open(os.path.join(self.low_res_path, self.low_res_images[idx])).convert('RGB')
        high_img = Image.open(os.path.join(self.high_res_path, self.high_res_images[idx])).convert('RGB')

        # Apply transforms
        low_img = self.transforms(low_img)
        high_img = self.transforms(high_img)

        return low_img, high_img



