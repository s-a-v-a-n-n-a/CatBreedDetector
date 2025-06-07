from test import main as test

import fire

from infer import main as infer
from infer_onnx import main as infer_onnx
from infer_triton import main as infer_triton
from infer_trt import main as infer_trt
from train import main as train


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "test": test,
            "infer": {
                "from_checkpoint": infer,
                "onnx": infer_onnx,
                "tensorrt": infer_trt,
                "triton": infer_triton,
            },
        }
    )
