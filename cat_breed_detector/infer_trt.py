import fire
import json
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import typing as tp
import numpy as np
from PIL import Image
from pathlib import Path
from hydra import compose, initialize

from utilities.model_getter import get_processor
from utilities.data_handler import ensure_data_unpacked


MODEL_PATH = "models/model.fp32.trt"


class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        self.tensor_info = self._setup_io_info()
        self.name_to_idx = {name: i for i, name in enumerate(self.tensor_info.keys())}

    def _setup_io_info(self) -> tp.Dict[str, tp.Dict[str, tp.Union[str, tuple]]]:
        tensor_info = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            tensor_info[name] = {
                'shape': self.engine.get_tensor_shape(name),
                'dtype': self.engine.get_tensor_dtype(name),
                'mode': self.engine.get_tensor_mode(name)
            }
        return tensor_info
        
    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(f.read())
    
    def infer(self, inputs: tp.Dict[str, np.ndarray]) -> tp.Dict[str, np.ndarray]:
        stream = cuda.Stream()
        output_buffers = {}
        outputs = {}

        for name, data in inputs.items():
            if name not in self.tensor_info:
                raise ValueError(f"Tensor {name} was not found in model")

            input_mem = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod_async(input_mem, data, stream)

            self.context.set_tensor_address(name, int(input_mem))

        for name, info in self.tensor_info.items():
            if info['mode'] == trt.TensorIOMode.OUTPUT:
                shape = info['shape']
                dtype = trt.nptype(info['dtype'])
                output = np.empty(shape, dtype=dtype)

                output_mem = cuda.mem_alloc(output.nbytes)

                self.context.set_tensor_address(name, int(output_mem))

                output_buffers[name] = (output, output_mem)
                outputs[name] = output

        self.context.execute_async_v3(stream_handle=stream.handle)

        for name, (output, output_mem) in output_buffers.items():
            cuda.memcpy_dtoh_async(output, output_mem, stream)

        stream.synchronize()

        for output, output_mem in output_buffers.values():
            output_mem.free()

        return outputs


def main(
        images_to_analyze: Path
) -> None:
    with initialize(config_path="../configs", version_base="1.1"):
        config = compose(config_name="config")
        ensure_data_unpacked(MODEL_PATH)
        trt_model = TensorRTInference(MODEL_PATH)
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

            outputs = trt_model.infer(input_data)
            predicted_class = np.argmax(outputs["logits"][0])

            print(
                "For image: {0}, got breed: {1}".format(
                    image_path,
                    class_names[str(predicted_class)]
                )  
            )

if __name__ == "__main__":
    fire.Fire(main)
