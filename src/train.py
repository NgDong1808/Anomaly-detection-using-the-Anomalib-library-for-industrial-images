from anomalib.engine import Engine
from anomalib.models import Patchcore, Cfa
from anomalib.metrics import Evaluator, F1Score, AnomalibMetric
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from anomalib.callbacks import ModelCheckpoint
import torch
class Precision(AnomalibMetric, BinaryPrecision):
    pass
class Recall(AnomalibMetric, BinaryRecall):
    pass

def train(datamodule = None, type_model ="Patchcore", backbone ="resnet18", device ="cpu", lr = 0.01, max_epochs = 1):
    datamodule = datamodule
    checkpoint_callback = ModelCheckpoint(dirpath="./model", filename="best", enable_version_counter = False, save_last=False, save_top_k=1)
    test_metrics = [
        F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
        Precision(fields=["pred_label", "gt_label"], prefix="image_"),
        Recall(fields=["pred_label", "gt_label"], prefix="image_"),
    ]
    if(type_model == "Patchcore"):      
        model = Patchcore(num_neighbors=3)
        engine = Engine(callbacks=[checkpoint_callback])
    elif(type_model =="Cfa"):
        class MyCFA(Cfa):
            def configure_optimizers(self):
                return torch.optim.AdamW(
                    params=self.model.parameters(),
                    lr=lr,
                    weight_decay=5e-4,
                    amsgrad=True,
                )
        model = MyCFA(backbone=backbone)
        engine = Engine(callbacks=[checkpoint_callback], accelerator=device, max_epochs = max_epochs)

    model.evaluator = Evaluator(test_metrics=test_metrics)
    engine.fit(datamodule=datamodule, model=model)

    test_results = engine.test(datamodule=datamodule, model=model)
    return test_results

