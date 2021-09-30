import torch
import tensorrt as trt
from mmcv.tensorrt.tensorrt_utils import (torch_dtype_from_trt,
                                          torch_device_from_trt)

def inference_tensorrt(ckpt_path, distributed, data_loader, batch_size):
    # load engine
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(ckpt_path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    # For now, only support fixed input tensor
    cur_batch_size = engine.get_binding_shape(0)[0]

    print(cur_batch_size)
    assert batch_size == cur_batch_size, \
        ('Dataset and TensorRT model should share the same batch size, '
         f'but get {batch_size} and {cur_batch_size}')

    context = engine.create_execution_context()

    # get output tensor
    dtype = torch_dtype_from_trt(engine.get_binding_dtype(1))
    shape = tuple(context.get_binding_shape(1))
    device = torch_device_from_trt(engine.get_location(1))
    output = torch.empty(
        size=shape, dtype=dtype, device=device, requires_grad=False)

ckpt_path = '/home/ruiming/workspace/pro/source/mmaction2/work_dirs/fatigue_r50_clean/epoch_58.pth'

inference_tensorrt(ckpt_path, False, None, 1)