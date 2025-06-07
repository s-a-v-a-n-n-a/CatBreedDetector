import json
from pathlib import Path

import fire
import numpy as np
import tritonclient.http as httpclient
from hydra import compose, initialize
from PIL import Image

from utilities.model_getter import get_processor


def main(url: str, images_to_analyze: Path):
    with initialize(config_path="../configs", version_base="1.1"):
        config = compose(config_name="config")
        with open(config["data_loading"]["id2labels_meta"]) as labels_meta_file:
            class_names = json.load(labels_meta_file)
        client = httpclient.InferenceServerClient(url=url)

        processor = get_processor()

        for image_path in Path(images_to_analyze).glob("**/*"):
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            input_data = inputs["pixel_values"].numpy().astype(np.float32)
            inputs = httpclient.InferInput("pixel_values", input_data.shape, "FP32")
            inputs.set_data_from_numpy(input_data)

            outputs = [httpclient.InferRequestedOutput("logits")]
            response = client.infer(
                model_name="cat_breed_detector", inputs=[inputs], outputs=outputs
            )

            result = response.as_numpy("logits")
            predicted_class = np.argmax(result[0])

            print(
                "For image: {0}, got breed: {1}".format(
                    image_path, class_names[str(predicted_class)]
                )
            )


if __name__ == "__main__":
    fire.Fire(main)
