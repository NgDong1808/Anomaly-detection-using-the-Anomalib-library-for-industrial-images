
from anomalib.data.datasets.base import AnomalibDataset
from pathlib import Path
import pandas as pd

class MyCustomDataset(AnomalibDataset):
    def __init__(self, root, category, transform=None, split=None):
        super().__init__()
        self.transform = transform
        self.root = Path(root)
        self.category = category
        self.split = split
        self.samples = self._make_dataset()

    def _make_dataset(self) -> pd.DataFrame:
        samples_list = []
        split_path = self.root / self.split

        if self.split == "train":
            normal_path = split_path / "normal"
            if normal_path.exists():
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    for img_path in normal_path.glob(ext):
                        samples_list.append({
                            "image_path": str(img_path),
                        "mask_path": "",
                        "label": "normal",
                        "label_index": 0,
                        "split": self.split
                    })

        if self.split == "val":
            abnormal_path = split_path / "anomaly"
            normal_path = split_path / "normal"
            if normal_path.exists():
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    for img_path in normal_path.glob(ext):
                        samples_list.append({
                            "image_path": str(img_path),
                        "mask_path": "",
                        "label": "normal",
                        "label_index": 0,
                        "split": self.split
                    })
            if abnormal_path.exists():
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    for img_path in abnormal_path.glob(ext):
                        samples_list.append({
                            "image_path": str(img_path),
                            "mask_path": "",
                            "label": "anomaly",
                            "label_index": 1,
                            "split": self.split
                        })
        if(self.split == "test"):
            abnormal_path = split_path / "anomaly"
            normal_path = split_path / "normal"
            if normal_path.exists():
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    for img_path in normal_path.glob(ext):
                        samples_list.append({
                            "image_path": str(img_path),
                        "mask_path": "",
                        "label": "normal",
                        "label_index": 0,
                        "split": self.split
                    })
            if abnormal_path.exists():
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    for img_path in abnormal_path.glob(ext):
                        samples_list.append({
                            "image_path": str(img_path),
                            "mask_path": "",
                            "label": "anomaly",
                            "label_index": 1,
                            "split": self.split
                        })
        samples = pd.DataFrame(samples_list)
        samples.attrs["task"] = "classification"
        return samples

