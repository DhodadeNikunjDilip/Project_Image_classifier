from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        
        self._setup_classes()
        self._load_samples()
        
        if not self.samples:
            raise ValueError(f"No valid images found in {root_dir}")

    def _setup_classes(self):
        self.classes = sorted([d.name for d in os.scandir(self.root_dir) if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def _load_samples(self):
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for file in os.listdir(cls_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(cls_dir, file),
                        self.class_to_idx[cls]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
def custom_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):  # Changed num_workers to 0
    """Create a DataLoader with Windows-friendly settings"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )