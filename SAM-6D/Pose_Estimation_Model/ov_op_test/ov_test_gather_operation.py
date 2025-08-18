import os
import torch
from torch import nn
import numpy as np
import time
import openvino as ov
from xml.etree import ElementTree as ET

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pointnet2 import _ext  # PyTorch extension

# Paths
torch_model_path = "./test_model/gather_op.pth"
onnx_model_path = "./test_model/gather_op.onnx"
ov_model_path = "./test_model/gather_op.xml"
ov_kernel_path = "./ov_op/gather_operation_cl.xml"
ov_extension_lib_path = "./ov_op/build/libopenvino_operation_extension.so"

'''
# Dummy inputs
DUMMY_B = 2 
DUMMY_C = 64
DUMMY_N = 128
DUMMY_NPOINT = 32
'''

DUMMY_B = 7        # Batch size
DUMMY_C = 256      # Number of feature channels
DUMMY_N = 2049     # Total input points in 'features'
DUMMY_NPOINT = 196 # Number of gathered points (from idx)


# OpenVINO Core
core = ov.Core()
core.add_extension(ov_extension_lib_path)

if not os.path.exists(onnx_model_path):
    os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

### ----------------- PyTorch Custom Op Wrapper ----------------- ###
class GatherOpWrapper(torch.autograd.Function):
    @staticmethod
    def symbolic(g, features, idx):
        return g.op("GatherOperation", features, idx)

    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        if idx.dtype != torch.int32:
            idx = idx.to(torch.int32)

        _, C, N = features.size()
        ctx.for_backwards = (idx, C, N)

        return _ext.gather_points(features, idx) 


### ----------------- PyTorch Model Wrapper ----------------- ###
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.gather_op = GatherOpWrapper.apply

    def forward(self, features, idx):
        return self.gather_op(features, idx)


### ----------------- Input Data Setup ----------------- ###
def get_input_data():
    features = torch.randn(DUMMY_B, DUMMY_C, DUMMY_N, dtype=torch.float32)
    idx = torch.randint(0, DUMMY_N, (DUMMY_B, DUMMY_NPOINT), dtype=torch.int32)
    idx_int32 = idx.to(torch.int32)

    onnx_input = (features, idx)
    onnx_input_name = ["features", "idx"]

    ov_input = {
        "features": features.numpy(), 
        "idx": idx_int32.numpy()
    }
    ov_input_name = {"features": [DUMMY_B, DUMMY_C, DUMMY_N],
                     "idx": [DUMMY_B, DUMMY_NPOINT]}

    #print("features sample:", features[0, :, :5])  # prints first 5 points of the first sample
    #print("idx sample:", idx[0, :5, :])            # prints first 5 query groups
    return onnx_input, onnx_input_name, ov_input, ov_input_name


### ----------------- Export to ONNX ----------------- ###
def onnx_model_convert(model, onnx_input, onnx_input_name):
    with torch.no_grad():
        torch.onnx.export(
            model,
            onnx_input,
            onnx_model_path,
            opset_version=20,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            input_names=onnx_input_name,
            dynamic_axes = {k: {0: "batch"} for k in onnx_input_name},
            do_constant_folding=False,
            verbose=False,  # True , for detailed output
            export_params=True,
            keep_initializers_as_inputs=False
        )
    print(f"[ONNX] model export success: {onnx_model_path}")


### ----------------- Convert to OpenVINO ----------------- ###
def convert_to_openvino(ov_input, ov_input_name):
    ov_model = ov.convert_model(onnx_model_path,
                                input=ov_input_name,
                                example_input=ov_input,
                                extension=ov_extension_lib_path)
    ov.save_model(ov_model, ov_model_path)
    print(f"[OpenVINO] model convert success: {ov_model_path}")


### ----------------- CPU-Pytorch Inference -----------------###
def pytorch_infer(ov_input, device="CPU"):
    # PyTorch
    model = MyModel()
    model.load_state_dict(torch.load(torch_model_path, weights_only=True))
    model.eval()
    torch_start = time.time()
    #torch_out = model(torch.tensor(ov_input["features"]),torch.tensor(ov_input["idx"], dtype=torch.int32))    
    
    features_tensor = ov_input["features"].clone().detach()
    idx_tensor = ov_input["idx"].clone().detach().to(torch.int32)
    
    #features_tensor = torch.tensor(ov_input["features"])
    #idx_tensor = torch.tensor(ov_input["idx"], dtype=torch.int32)

    torch_out = model(features_tensor, idx_tensor) 
    torch_time = time.time() - torch_start
    print(f"Performance Capture:")
    print(f"[PYTORCH] infer time:      {torch_time * 1000:.2f} ms")
    return torch_out


### ----------------- OV CPU+GPU Inference -----------------###
def ov_infer(ov_input, device="CPU"):
    if device == "GPU":
        core.set_property("GPU", {"INFERENCE_PRECISION_HINT": "f32"})
        core.set_property("GPU", {"CONFIG_FILE": ov_kernel_path})

    # OpenVINO
    ov_model = core.read_model(ov_model_path)
    compiled_model = core.compile_model(ov_model, device)
    # warmup
    if device == "GPU":
        _ = compiled_model.infer_new_request(ov_input)  
        
    ov_start = time.time()
    result = compiled_model.infer_new_request(ov_input)
    ov_time = time.time() - ov_start

    ov_out = list(result.values())[0]
    print(f"[OpenVINO] {device} infer time: {ov_time * 1000:.2f} ms")
    return ov_out


### ----------------- Inference result + Comparison ----------------- ###
def compare_infer(torch_out, ov_out, device):
    # Compare
    print("Torch output shape:", torch_out.shape)
    ov_tensor = torch.from_numpy(ov_out)
    print("OV output shape:", ov_tensor.shape)
    diff = torch.abs(ov_tensor - torch_out)
    print(f"[OV {device}] + pytorch  Max diff: {diff.max()}")
    print(f"[OV {device}] + pytorch  Min diff: {diff.min()}")
    print(f"[OV {device}] + pytorch  MSE: {(diff ** 2).mean()}")
    

    #print("Torch output stats: min =", torch_out.min().item(), ", max =", torch_out.max().item(), ", mean =", torch_out.mean().item())
    #print("OV output stats: min =", ov_tensor.min().item(), ", max =", ov_tensor.max().item(), ", mean =", ov_tensor.mean().item())

    assert torch.allclose(ov_tensor, torch_out, atol=1e-4)
    print(f"[COMPARE Result {device}] and Pytorch PASSED")


### ----------------- Main ----------------- ###
def main():
    np.random.seed(324)
    torch.manual_seed(32)

    model = MyModel()
    model.eval()
    torch.save(model.state_dict(), torch_model_path)

    onnx_input, onnx_input_name, ov_input, ov_input_name = get_input_data()

    onnx_model_convert(model, onnx_input, onnx_input_name)
    
    convert_to_openvino(ov_input, ov_input_name)

##--------- compare CPU pytorch vs OV GPU -------- ###
    print(f"Gather Operation:")
    torch_out = pytorch_infer(ov_input)
    ov_out = ov_infer(ov_input, "GPU")
    compare_infer(torch_out, ov_out, "GPU")
    
##--------- compare CPU pytorch vs OV CPU -------- ###
    #torch_out = pytorch_infer(ov_input)
    #ov_out = ov_infer(ov_input) #CPU
    #compare_infer(torch_out, ov_out, "CPU")

if __name__ == "__main__":
    main()
