import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    Spacingd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    DivisiblePadd
)

class CirrMRI3DDataset(Dataset):
    def __init__(self, base_dir, modality, split='train', transform=None):
        """
        Args:
            base_dir (str): Base directory path, e.g. "/content/drive/MyDrive/cirrmri/CirrMRI600+"
            modality (str): Either "Cirrhosis_T1_3D" or "Cirrhosis_T2_3D"
            split (str): One of 'train', 'valid', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform

        # Construct the directories for images and masks.
        # Example: .../Cirrhosis_T1_3D/Cirrhosis_T1_3D/train_images
        self.image_dir = os.path.join(base_dir, modality, modality, f"{split}_images")
        self.mask_dir = os.path.join(base_dir, modality, modality, f"{split}_masks")

        # List all nii.gz files in the images directory (assuming masks have the same names)
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.nii.gz')])
        if not self.image_files:
            raise RuntimeError(f"No .nii.gz files found in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get the file name based on the index
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, img_filename)  # Assuming same filename for mask

        # Load the image and mask using nibabel
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        img_array = img_nii.get_fdata()
        mask_array = mask_nii.get_fdata()

        # Optionally: Convert the numpy arrays to PyTorch tensors.
        # Note: For segmentation masks, it's common to use dtype=torch.long.
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_array, dtype=torch.long)

        # Add a channel dimension if your network expects one.
        # For a 3D image, if the shape is (H, W, D), then unsqueeze to get (1, H, W, D).
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        sample = {'image': img_tensor, 'mask': mask_tensor, 'id': int(os.path.splitext(os.path.basename(img_filename))[0])}

        # Apply transformation if provided
        if self.transform:
            sample = self.transform(sample)

        return sample

train_transforms = Compose([
    EnsureChannelFirstd(keys=['image', 'mask']),
    Spacingd(keys=['image', 'mask'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
    RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),
    RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1),
    RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=2),
    RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.5),
    NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
    DivisiblePadd(keys=['image', 'mask'], k=16),
    ToTensord(keys=['image', 'mask'])
])

# For validation (NOT TESTING), only apply deterministic preprocessing.
val_transforms = Compose([
    EnsureChannelFirstd(keys=['image', 'mask']),
    Spacingd(keys=['image', 'mask'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
    NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
    ToTensord(keys=['image', 'mask'])
])


def get_dataloaders(base_dir, modality, train_batch_size=2, valid_batch_size=2, test_batch_size=2):
    """
    Utility function to create dataloaders for train, valid, and test splits.
    """
    train_dataset = CirrMRI3DDataset(base_dir, modality, split='train', transform=train_transforms)
    valid_dataset = CirrMRI3DDataset(base_dir, modality, split='valid', transform=val_transforms)
    test_dataset  = CirrMRI3DDataset(base_dir, modality, split='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader

# Example usage:
# base_dir = "/content/drive/MyDrive/cirrmri/CirrMRI600+"
# modality = "Cirrhosis_T1_3D"  # or "Cirrhosis_T2_3D"
# train_loader, valid_loader, test_loader = get_dataloaders(base_dir, modality)
