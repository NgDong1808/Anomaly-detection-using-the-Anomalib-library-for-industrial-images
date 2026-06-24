from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from anomalib.data.dataclasses import ImageBatch
from src.dataset import MyCustomDataset

class MyDataModule(LightningDataModule):
    def __init__(self, root: str = "./datasets", category: str = "delicious",
        train_batch_size: int = 1, eval_batch_size: int = 1, num_workers: int = 2, transform = None):
        self.root = root
        self.name = "custom_dataset"
        self.category = category
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.allow_zero_length_dataloader_with_multiple_devices = False

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:

            self.train_dataset = MyCustomDataset(
            root=self.root,
            category=self.category,
            transform=self.transform,
            split="train"
            )
            self.val_dataset = MyCustomDataset(
            root=self.root,
            category=self.category,
            transform=self.transform,
            split="val"
            )
        if stage == "test" or stage is None:
            self.test_dataset = MyCustomDataset(
            root=self.root,
            category=self.category,
            transform=self.transform,
            split="test"
            )

    def train_dataloader(self):
        return DataLoader(
        dataset=self.train_dataset,
        batch_size=self.train_batch_size,
        shuffle=True,
        num_workers=self.num_workers,
        collate_fn=ImageBatch.collate 
        )
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ImageBatch.collate
        )
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ImageBatch.collate
        )
    
   