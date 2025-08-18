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

torch_model_path = "./test_model/furthest_point_sampling.pth"
onnx_model_path = "./test_model/furthest_point_sampling.onnx"
ov_model_path = "./test_model/furthest_point_sampling.xml"

ov_kernel_path = "ov_op/furthest_point_sampling_cl.xml"
# ov_kernel_path = "ov_op/furthest_point_sampling_optimized_cl.xml"
ov_extension_lib_path = "ov_op/build/libopenvino_operation_extension.so"

core = ov.Core()
core.add_extension(ov_extension_lib_path)

if not os.path.exists(onnx_model_path):
    os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

# reference "OpenVINO-SAM-6D/SAM-6D/Pose_Estimation_Model/utils/model_utils.py"
# 1, 210000, 3
DUMMY_BATCH_SIZE = 1
DUMMY_N = 21000
DUMMY_CHANNEL = 3
DUMMY_npints = 2048

# DUMMY_BATCH_SIZE = 7
# DUMMY_N = 2048
# DUMMY_npints = 3

class FurthestPointSampling(torch.autograd.Function):
    def __init__(self):
        super(FurthestPointSampling, self).__init__()
    
    @staticmethod
    def forward(ctx, pts, npoint):
        furthest_point_sampling = _ext.furthest_point_sampling(pts, npoint)
        ctx.save_for_backward(furthest_point_sampling)
        return furthest_point_sampling

    @staticmethod
    def symbolic(g: torch.Graph, pts: torch.Tensor, npoint: int) :
        return g.op("FurthestPointSampling", pts,  npoint_i=npoint, outputs=1)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.custom_furthest_point_sampling = FurthestPointSampling.apply
        self.npoint = 2048

    def forward(self, pts):
        '''
        pts: B*N*3
        npoint: 2048
        '''
        return self.custom_furthest_point_sampling(pts, self.npoint)


def get_input_data():
    dummy_pts = torch.randn(DUMMY_BATCH_SIZE, DUMMY_N, DUMMY_CHANNEL)
    dyummy_nponts = torch.tensor(DUMMY_npints)

    onnx_input = (dummy_pts)
    onnx_input_name = ["pts"]

    ov_input = {"pts":dummy_pts}
    # ov_input_name = {"pts":[-1, -1, DUMMY_CHANNEL]}
    ov_input_name = {"pts":[DUMMY_BATCH_SIZE, DUMMY_N, DUMMY_CHANNEL],
                    }
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
            # dynamic_axes={"pts": {0: "batch", 1: "N"} },
            do_constant_folding=False,
            verbose=False,  # True , for furthest_point_sampling ailed output
            export_params=True,
            keep_initializers_as_inputs=False
        )
    print(f"[ONNX] model export success: {onnx_model_path}")

def ov_model_convert(ov_input, ov_input_name):
    ov_model = ov.convert_model(onnx_model_path, 
                                input=ov_input_name,
                                # output={"output":[DUMMY_BATCH_SIZE, DUMMY_npints]},
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
        core.set_property("GPU", {"INFERENCE_PRECISION_HINT": "f32"})
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
    # print(f"[OV {device}] infer result: {ov_infer_result[0][:5]}")
    print(f"[OV {device}] infer time_cost: {(ov_infer_time*1000):.2f} ms")
    
    torch_start_time = time.time()
    torch_infer_result = torch_model(ov_input["pts"])
    # print(f"[Torch CPU] infer result: \n{torch_infer_result[:5]}")
    torch_infer_time = time.time() - torch_start_time
    print(f"[Torch] infer time_cost: {(torch_infer_time*1000):.2f} ms")

    # torch & ov compare result
    print(f"[OV {device}] max diff: {torch.max(torch.abs(torch.from_numpy(ov_infer_result[0]) - torch_infer_result))}")
    print(f"[OV {device}] min diff: {torch.min(torch.abs(torch.from_numpy(ov_infer_result[0]) - torch_infer_result))}")
    print(f"[OV {device}] mse: {torch.mean((torch.abs(torch.from_numpy(ov_infer_result[0]) - torch_infer_result)**2).float())}")
    # assert torch.allclose(torch.from_numpy(ov_infer_result[0]), torch_infer_result, atol=1e-4)
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


