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

torch_model_path = "./test_model/ball_query.pth"
onnx_model_path = "./test_model/ball_query.onnx"
ov_model_path = "./test_model/ball_query.xml"

# ov_kernel_path = "../model/ov_pointnet2_op/ball_query.xml"
# ov_extension_lib_path = "../model/ov_pointnet2_op/build/libopenvino_operation_extension.so"
ov_kernel_path = "ov_op/ball_query_cl.xml"
ov_extension_lib_path = "ov_op/build/libopenvino_operation_extension.so"

DUMMY_BATCH_SIZE = 1
DUMMY_NPOINT = 1024
DUMMY_N = 256
DUMMY_NSAMPLE = 64

core = ov.Core()
core.add_extension(ov_extension_lib_path)

if not os.path.exists(onnx_model_path):
    os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)


# reference: OpenVINO-SAM-6D/SAM-6D/Pose_Estimation_Model/model/pointnet2/pointnet2_utils.py
class BallQueryWrapper(torch.autograd.Function):
    @staticmethod
    def symbolic(g, new_xyz, xyz, radius, nsample):
        return g.op("BallQuery", new_xyz, xyz, radius_f=radius, nsample_i=nsample)

    # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
    @staticmethod
    def forward(ctx, new_xyz, xyz, radius, nsample):
        inds = _ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.ball_query = BallQueryWrapper.apply
        self.radius = 0.1
        self.nsample = DUMMY_NSAMPLE

    def forward(self, new_xyz, xyz):
        xyz_n = xyz.shape[1]
        # xyz_n = torch.tensor(xyz.shape[1], dtype=torch.int32, device=xyz.device)
        # xyz : torch.Tensor
        #     xyz coordinates of the features (B, N, 3)
        # new_xyz : torch.Tensor
        #     centriods (B, npoint, 3)
        return self.ball_query(new_xyz, xyz, self.radius, self.nsample) # (B, npoint, nsample)

def get_input_data():
    dummy_new_xyz = torch.randn(DUMMY_BATCH_SIZE, DUMMY_NPOINT, 3)
    dummy_xyz = torch.randn(DUMMY_BATCH_SIZE, DUMMY_N, 3)

    onnx_input = (dummy_new_xyz, dummy_xyz)
    onnx_input_name = ["new_xyz", "xyz"]

    ov_input = {"new_xyz":dummy_new_xyz,
                "xyz":dummy_xyz}
    ov_input_name = {"new_xyz":[DUMMY_BATCH_SIZE, DUMMY_NPOINT, 3],
                     "xyz":[DUMMY_BATCH_SIZE, DUMMY_N, 3],  }
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


def torch_ov_compare_cpu(ov_input, device="CPU"):
    print("--------------torch & ov compare result------------------")
    # Init model
    if device == "GPU":
        core.set_property("GPU", {"CONFIG_FILE": ov_kernel_path})

    ov_model = core.read_model(ov_model_path)
    compiled_model = core.compile_model(ov_model, device)

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
    # print(f"[OpenVINO] infer result: {ov_infer_result[0][:5]}")
    print(f"[OpenVINO] infer time_cost: {(ov_infer_time*1000):.2f} ms")
    
    torch_start_time = time.time()
    torch_infer_result = torch_model(ov_input["new_xyz"], ov_input["xyz"])
    # print(f"[Pytorch] infer result: \n{torch_infer_result[:5]}")
    torch_infer_time = time.time() - torch_start_time
    print(f"[Torch] infer time_cost: {(torch_infer_time*1000):.2f} ms")

    # torch & ov compare result
    print(f"[OV {device}] max diff: {torch.max(torch.abs(torch.from_numpy(ov_infer_result[0]) - torch_infer_result))}")
    print(f"[OV {device}] min diff: {torch.min(torch.abs(torch.from_numpy(ov_infer_result[0]) - torch_infer_result))}")
    print(f"[OV {device}] mse: {torch.mean((torch.abs(torch.from_numpy(ov_infer_result[0]) - torch_infer_result)**2).float())}")
    assert torch.allclose(torch.from_numpy(ov_infer_result[0]), torch_infer_result, atol=1e-4)
    print(f"[OV {device}] torch & ov infer result compare success")



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

    torch_ov_compare_cpu(ov_input, "CPU")
    torch_ov_compare_cpu(ov_input, "GPU")
    # ov_infer(ov_input, "GPU")


if __name__ == "__main__":
    main()







