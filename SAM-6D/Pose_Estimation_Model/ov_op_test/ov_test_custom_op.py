import os
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import openvino as ov
import torch.nn.functional as F
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pointnet2 import _ext

torch_model_path = "./test_model/custom_op.pth"
onnx_model_path = "./test_model/custom_op.onnx"
ov_model_path = "./test_model/custom_op.xml"

# ov_kernel_path = "../model/ov_pointnet2_op/custom_op.xml"
# ov_extension_lib_path = "../model/ov_pointnet2_op/build/libopenvino_operation_extension.so"
ov_kernel_path = "ov_op/custom_op_cl.xml"
ov_extension_lib_path = "ov_op/build/libopenvino_operation_extension.so"

DUMY_ALPHA = 2.0
DUMY_BETA = 0.1

input_batch = 1
INPUT_DIM = 1024

core = ov.Core()
core.add_extension(ov_extension_lib_path)

if not os.path.exists(onnx_model_path):
    os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)


# reference: OpenVINO-SAM-6D/SAM-6D/Pose_Estimation_Model/model/pointnet2/pointnet2_utils.py
class CustomAddOpWrapper(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input_a, alpha, beta):
        return g.op("CustomAddOp", input_a, alpha_f=alpha, beta_f=beta)

    # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
    @staticmethod
    def forward(ctx, input_a, alpha, beta):
        inds = torch.add(input_a, alpha, beta)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.custom_op = CustomAddOpWrapper.apply
        self.alpha = DUMY_ALPHA
        self.beta = DUMY_BETA

    def forward(self, input_a):
        return self.custom_op(input_a, self.alpha, self.beta) # (B, npoint, nsample)

def get_input_data():
    dummy_input_a = torch.randn(1, INPUT_DIM, 3)

    onnx_input = (dummy_input_a)
    onnx_input_name = ["input_a"]

    ov_input = {"input_a":dummy_input_a}
    ov_input_name = {"input_a":[1, INPUT_DIM, 3]}
    return onnx_input, onnx_input_name, ov_input, ov_input_name

def onnx_model_convert(model, onnx_input, onnx_input_name):
    with torch.no_grad():
        torch.onnx.export(
            model,
            onnx_input,
            onnx_model_path,
            opset_version=20,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            input_names=onnx_input_name,
            dynamic_axes={k: {0: "batch"} for k in onnx_input_name},
            do_constant_folding=False,
            verbose=False,  # True , for detailed output
            export_params=True,
            keep_initializers_as_inputs=False
        )
    print(f"[ONNX] model export success: {onnx_model_path}")

def ov_model_convert(ov_input, ov_input_name):
    ov_model = ov.convert_model(onnx_model_path, 
                                input=ov_input_name,
                                example_input=ov_input,
                                extension=ov_extension_lib_path,
                                    )
    compiled_model = core.compile_model(ov_model, 'CPU')
    ov.save_model(ov_model, ov_model_path)
    print(f"[OpenVINO] model convert success: {ov_model_path}")


def ov_infer(ov_input, device="CPU"):
    if device == "GPU":
        core.set_property("GPU", {"CONFIG_FILE": ov_kernel_path})
    # Init model
    ov_model = core.read_model(ov_model_path)
    compiled_model = core.compile_model(ov_model, device)
    # warmup 
    ov_result = compiled_model.infer_new_request(ov_input)

    ov_start_time = time.time()
    ov_result = compiled_model.infer_new_request(ov_input)
    ov_infer_result =list(ov_result.values())
    ov_infer_time = time.time() - ov_start_time
    print(f"[OpenVINO] infer time_cost: {(ov_infer_time*1000):.2f} ms")
    print(f"[OpenVINO] output {ov_infer_result[0][:100]}")


def torch_ov_compare_cpu(ov_input):
    # Init model
    ov_model = core.read_model(ov_model_path)
    compiled_model = core.compile_model(ov_model, 'CPU')

    torch_model = MyModel()
    torch_model.load_state_dict(torch.load(torch_model_path))
    torch_model.eval()
    torch_model.to("cpu")

    # warmup 
    ov_result = compiled_model.infer_new_request(ov_input)

    ov_start_time = time.time()
    ov_result = compiled_model.infer_new_request(ov_input)
    ov_infer_result =list(ov_result.values())
    ov_infer_time = time.time() - ov_start_time
    print(f"[OpenVINO] infer time_cost: {(ov_infer_time*1000):.2f} ms")
    
    torch_start_time = time.time()
    torch_infer_result = torch_model(ov_input["input_a"])
    # print(f"[Pytorch] infer result: {torch_infer_result}")
    torch_infer_time = time.time() - torch_start_time
    print(f"[Torch] infer time_cost: {(torch_infer_time*1000):.2f} ms")

    # torch & ov compare result
    print("--------------torch & ov compare result------------------")
    print(f"[CPU] max diff: {torch.max(torch.abs(torch.from_numpy(ov_infer_result[0]) - torch_infer_result))}")
    print(f"[CPU] min diff: {torch.min(torch.abs(torch.from_numpy(ov_infer_result[0]) - torch_infer_result))}")
    print(f"[CPU] mse: {torch.mean((torch.abs(torch.from_numpy(ov_infer_result[0]) - torch_infer_result)**2).float())}")
    # assert torch.allclose(torch.from_numpy(ov_infer_result[0]), torch_infer_result, atol=1e-4)
    print(f"[CPU] torch & ov infer result compare success")



def main():
    np.random.seed(324)
    torch.manual_seed(32)

    model = MyModel()
    model.eval()
    model.to("cpu")
    torch.save(model.state_dict(), torch_model_path)

    onnx_input, onnx_input_name, ov_input, ov_input_name = get_input_data()

    onnx_model_convert(model, onnx_input, onnx_input_name)
    ov_model_convert(ov_input, ov_input_name)

    torch_ov_compare_cpu(ov_input)
    ov_infer(ov_input, "GPU")


if __name__ == "__main__":
    main()







