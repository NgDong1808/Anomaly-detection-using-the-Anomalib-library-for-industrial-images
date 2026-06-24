from fastapi import FastAPI
from torchvision.transforms.v2 import Resize
from src.datamodule import MyDataModule
from src.train import train
from src.test import inference
from anomalib.data import PredictDataset
import uvicorn
import torch

app = FastAPI()
state = {
    "device" : None,
    "training": False,
    "trained": False,
    "img_width": 300,
    "img_height": 150,
    "root" : None,
    "type_model":"Patchcore"
}


@app.get(
    "/status",
    tags=["Status"]
)
def status():
    """Return whether the model is trained/training and which device is available."""
    return {
        "trained":    state["trained"],
        "training":   state["training"],
        "device":     "cuda" if torch.cuda.is_available() else "cpu",
    }

@app.post(
    "/set_dataset_path",
    tags=["Dataset"],
    summary="Set dataset root path",
    description="""Set the root path for the dataset. The dataset should have two subdirectories: 'images' and 'masks""",
    )
def set_dataset_path(root: str):
    """Set the root path for the dataset. The dataset should have two subdirectories: 'images' and 'masks'."""
    if(state["training"] == False):
        state["root"] = root
        return {"message": "Dataset paths set successfully."}
    else:
        return {"message": "Is training."}

@app.post(
    "/train",
    tags=["Train"],
    description="""
    Choose 2 models: Patchcore or Cfa
    """,

)
def trainning(category="my_data", type_model="Patchcore", backbone="resnet18", lr:float=0.01, train_batch_size:int=2, eval_batch_size:int=2,  num_workers:int=2, w:int=256, h:int=256):
    state["training"] = True
    state["trained"] = False
    state["img_height"] = h
    state["img_width"] = w

    transform = Resize(size=( state["img_width"], state["img_height"]))
    datamodule = MyDataModule(root=state["root"], category=category, num_workers=num_workers, train_batch_size=train_batch_size, eval_batch_size= eval_batch_size, transform=transform)
    results = train(type_model=type_model, backbone=backbone, device=state["device"], lr=lr, datamodule=datamodule)

    message =[]
    for result in results:
        for key, value in result.items():
            tmp = f"{key}: {value}"
            message.append(tmp)
    state["type_model"] = type_model
    state["training"] = False
    state["trained"] = True
    return{
        "message": message
    }
@app.post(
    "/inference",
    tags=["inference"]
)
def infer(testPath:str):
    image_size =(state["img_width"], state["img_height"])
    dataset = PredictDataset(
        path=testPath,
        image_size=image_size,
    )
    inference(type_model=state["type_model"], dataset=dataset)
   
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8002)
