from anomalib.engine import Engine
from anomalib.models import Patchcore, Cfa
from pathlib import Path
from anomalib.visualization.image.item_visualizer import visualize_image_item
def inference(type_model ="Patchcore", dataset = None):

    if(type_model == "Patchcore"):      
        model = Patchcore(num_neighbors=3)
        engine = Engine()
    elif(type_model =="Cfa"):
        model = Cfa()
        engine = Engine()

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path="./model/best.ckpt",
    )

    save_dir = Path("./overlay_results")
    save_dir.mkdir(exist_ok=True)
    for batch_idx, batch in enumerate(predictions):
        batch_size = len(batch.pred_label)
        for i in range(batch_size):
            for item in batch:
                # Create overlay
                vis = visualize_image_item(
                    item,
                    text_config={"enable": False},
                    fields=["image", "anomaly_map"],
                    overlay_fields=[("image", ["anomaly_map"])],
                    overlay_fields_config={
                        "anomaly_map": {
                            "colormap": True,
                            "normalize": False,
                        }
                    }
                )
                file_name = Path(item.image_path).name + "_result.png"
                save_path = save_dir/file_name
                vis.save(save_path)
                print(f"Saved: {save_path}")
                