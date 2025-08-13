// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_operation.hpp"
#include <cmath>

using namespace TemplateExtension;

//! [op:ctor]
GatherOperation::GatherOperation(const ov::Output<ov::Node>& features, const ov::Output<ov::Node>& idx) : Op({features, idx}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]


//! [op:validate]
void GatherOperation::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    /*
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
    */
    const auto& features_input = input(0);
    const auto& idx_input = input(1);

    auto features_shape = features_input.get_partial_shape();  // [B, C, N]
    auto idx_shape = idx_input.get_partial_shape();            // [B, npoint, nsample]

    // Set output shape to 4D: [B, C, npoint, nsample]
    ov::PartialShape output_shape = {features_shape[0], features_shape[1], idx_shape[1], idx_shape[2]};
    set_output_type(0, features_input.get_element_type(), output_shape);
}

//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> GatherOperation::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    // OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

    return std::make_shared<GatherOperation>(new_args.at(0), new_args.at(1));
}
//! [op:copy]

//! [op:visit_attributes]
bool GatherOperation::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool GatherOperation::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float* features = inputs[0].data<const float>();
    const int* idx = inputs[1].data<const int>();

    int b = inputs[0].get_shape()[0]; // batch size
    int c = inputs[0].get_shape()[1]; // channels
    int n = inputs[0].get_shape()[2]; // number of points
    int npoints = inputs[1].get_shape()[1]; // number of points to gather
    int nsample = inputs[1].get_shape()[2];

    printf("\n------------------------------------------------------------->nmsamplep:%d", nsample);

    ov::PartialShape output_shape = {b, c, npoints, nsample};
    outputs[0].set_shape(output_shape.to_shape());
    auto* out_tensor = outputs[0].data<float>();
    
    // Initialize output to 0
    int output_total = b * c * npoints * nsample;
    for (int i = 0; i < output_total; ++i) {
	    out_tensor[i] = 0.0f;
    }
    for (int batch = 0; batch < b; ++batch) {
         for (int channel = 0; channel < c; ++channel) {
	     for (int point = 0; point < npoints; ++point) {
	          for (int sample = 0; sample < nsample; ++sample) {
	                int index = idx[batch * (npoints * nsample) + point * nsample + sample];
		 	if (index >= 0 && index < n) {
			    float val = features[batch * (c * n) + channel * n + index];
			    if (std::isnan(val) || std::isinf(val)) {
				val = 0.0f;
			    }
			    out_tensor[batch * (c * npoints * nsample) + channel * (npoints * nsample) + point * nsample + sample] = val;
			} else {
			    out_tensor[batch * (c * npoints * nsample) + channel * (npoints * nsample) + point * nsample + sample] = 0.0f;
		        }
	          }
	     }
         }
    } 

    /*
    // Debug: record input data
    const bool debug = false; // true / false
    if (debug) {
        // record features input data
        int features_total = b * c * n;
        FILE* fp_features = fopen("output/ov_gather_operation_input.txt", "a");
        if (fp_features) {
            fprintf(fp_features, "----- gather_operation features input -----\n");
            fprintf(fp_features, "Shape: B=%d, C=%d, N=%d\n", b, c, n);
            for (int i = 0; i < features_total; ++i) {
                fprintf(fp_features, "%f\n", features[i]);
            }
            fclose(fp_features);
        } else {
            std::cerr << "[GatherOperation Debug] Failed to open output/ov_gather_operation_input.txt for writing!" << std::endl;
        }
        
        // record idx input data
        int idx_total = b * npoints;
        FILE* fp_idx = fopen("output/ov_gather_operation_input.txt", "a");
        if (fp_idx) {
            fprintf(fp_idx, "----- gather_operation idx input -----\n");
            fprintf(fp_idx, "Shape: B=%d, npoints=%d\n", b, npoints);
            for (int i = 0; i < idx_total; ++i) {
                fprintf(fp_idx, "%d\n", idx[i]);
            }
            fclose(fp_idx);
        } else {
            std::cerr << "[GatherOperation Debug] Failed to open output/ov_gather_operation_input.txt for writing!" << std::endl;
        }
        
        // record output data
        std::cout << "[GatherOperation Debug] out_tensor: ";
        int total = b * c * npoints;
        float* out_data = out_tensor;
        // for (int i = 0; i < total; ++i) {
        //     std::cout << out_data[i] << ' ';
        // }
        std::cout << std::endl;
        // Save to file, for comparison with PyTorch
        FILE* fp = fopen("output/ov_gather_operation.txt", "a");
        if (fp) {
            fprintf(fp, "----- gather_operation call -----\n");
            for (int i = 0; i < total; ++i) {
                fprintf(fp, "%f ", out_data[i]);
                fprintf(fp, "\n");
            }
            fclose(fp);
        } else {
            std::cerr << "[GatherOperation Debug] Failed to open output/ov_gather_operation.txt for writing!" << std::endl;
        }
    }*/
    // out.set_shape(in.get_shape());
    // memcpy(out.data(), in.data(), in.get_byte_size());
    return true;
}

bool GatherOperation::has_evaluate() const {
    return true;
}
//! [op:evaluate]
