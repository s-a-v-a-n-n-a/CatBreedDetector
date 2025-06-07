import fire
import json
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path
from hydra import compose, initialize

from utilities.model_getter import get_processor
from utilities.data_handler import ensure_data_unpacked


MODEL_PATH = "models/model.onnx"


def main(
        images_to_analyze: Path
) -> None:
    with initialize(config_path="../configs", version_base="1.1"):
        config = compose(config_name="config")
        ensure_data_unpacked(MODEL_PATH)
        model = onnx.load(MODEL_PATH)
        ort_session = ort.InferenceSession(MODEL_PATH)
        with open(config["data_loading"]["id2labels_meta"]) as labels_meta_file:
            class_names = json.load(labels_meta_file)

        processor = get_processor()

        for image_path in Path(images_to_analyze).glob('**/*'):
            image = Image.open(image_path).convert("RGB")
            inputs = processor(
                images=image,
                return_tensors="pt"
            )
            input_data = {"pixel_values": inputs["pixel_values"].numpy()}

            outputs = ort_session.run(None, input_data)
            predicted_class = np.argmax(outputs[0])

            print(
                "For image: {0}, got breed: {1}".format(
                    image_path,
                    class_names[str(predicted_class)]
                )  
            )


if __name__ == "__main__":
    fire.Fire(main)
