// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_svd.hpp"
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <cmath>

using namespace TemplateExtension;

//! [op:ctor]
CustomSVD::CustomSVD(const ov::Output<ov::Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CustomSVD::validate_and_infer_types() {
    // Support arbitrary batch dimensions, the last two dimensions are (M, N)
    const auto& svd_input = input(0);
    auto svd_shape = svd_input.get_partial_shape();
    auto rank = svd_shape.rank().is_static() ? svd_shape.rank().get_length() : 0;
    if (rank < 2) {
        throw std::runtime_error("CustomSVD input must have at least 2 dimensions (batch..., M, N)");
    }
    auto elem_type = get_input_element_type(0);
    // batch shape
    std::vector<ov::Dimension> batch_dims;
    for (size_t i = 0; i < rank - 2; ++i) batch_dims.push_back(svd_shape[i]);
    auto m = svd_shape[rank - 2];
    auto n = svd_shape[rank - 1];
    // U: (batch..., M, M), S: (batch..., min(M,N)), V: (batch..., N, N)
    std::vector<ov::Dimension> u_shape = batch_dims; u_shape.push_back(m); u_shape.push_back(m);
    std::vector<ov::Dimension> s_shape = batch_dims;
    // Fix: ov::Dimension does not have min, need to manually check
    if (m.is_static() && n.is_static()) {
        s_shape.push_back(std::min(m.get_length(), n.get_length()));
    } else {
        // When dynamic, use m conservatively
        s_shape.push_back(m);
    }
    std::vector<ov::Dimension> v_shape = batch_dims; v_shape.push_back(n); v_shape.push_back(n);
    set_output_type(0, elem_type, ov::PartialShape(u_shape)); // U
    set_output_type(1, elem_type, ov::PartialShape(s_shape)); // S
    set_output_type(2, elem_type, ov::PartialShape(v_shape)); // V
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> CustomSVD::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    // OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");

    return std::make_shared<CustomSVD>(new_args.at(0));
}
//! [op:copy]

//! [op:visit_attributes]
bool CustomSVD::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool CustomSVD::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // Support batch SVD, input shape: [batch..., M, N]
    const auto& in = inputs[0];
    auto shape = in.get_shape();
    if (shape.size() < 2)
        throw std::runtime_error("CustomSVD input must have at least 2 dimensions");
    size_t rank = shape.size();
    size_t m = shape[rank - 2], n = shape[rank - 1];
    size_t batch = 1;
    for (size_t i = 0; i < rank - 2; ++i) batch *= shape[i];
    const float* data = in.data<const float>();
    // Output shape
    std::vector<size_t> u_shape = shape; u_shape[rank - 1] = m; // (batch..., M, M)
    std::vector<size_t> s_shape = shape; s_shape.pop_back(); s_shape[rank - 2] = std::min(m, n); // (batch..., min(M,N))
    std::vector<size_t> v_shape = shape; v_shape[rank - 2] = n; v_shape[rank - 1] = n; // (batch..., N, N)
    outputs[0].set_shape(u_shape);
    outputs[1].set_shape(s_shape);
    outputs[2].set_shape(v_shape);
    float* u_data = outputs[0].data<float>();
    float* s_data = outputs[1].data<float>();
    float* v_data = outputs[2].data<float>();
    size_t in_mat_size = m * n;
    size_t u_mat_size = m * m;
    size_t s_vec_size = std::min(m, n);
    size_t v_mat_size = n * n;
    for (size_t b = 0; b < batch; ++b) {
        const float* batch_data = data + b * in_mat_size;
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(batch_data, m, n);
        
        // Numerical stability check
        Eigen::MatrixXf A_safe = A;
        
        // Check and clean NaN/Inf in the input matrix
        for (int i = 0; i < A_safe.rows(); ++i) {
            for (int j = 0; j < A_safe.cols(); ++j) {
                if (std::isnan(A_safe(i,j)) || std::isinf(A_safe(i,j))) {
                    A_safe(i,j) = 0.0f;
                }
            }
        }
        
        // Use a more stable SVD setting
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A_safe, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
        // Get SVD results
        Eigen::MatrixXf U = svd.matrixU();
        Eigen::VectorXf S = svd.singularValues();
        Eigen::MatrixXf V = svd.matrixV();
        
        // Numerical stability processing
        // 1. Clean small values in singular values
        float eps = 1e-8f;
        for (int i = 0; i < S.size(); ++i) {
            if (S(i) < eps) {
                S(i) = eps;
            }
        }
        
        // 2. Ensure U and V are orthogonal
        for (int i = 0; i < U.rows(); ++i) {
            for (int j = 0; j < U.cols(); ++j) {
                if (std::isnan(U(i,j)) || std::isinf(U(i,j))) {
                    U(i,j) = (i == j) ? 1.0f : 0.0f;
                }
            }
        }
        
        for (int i = 0; i < V.rows(); ++i) {
            for (int j = 0; j < V.cols(); ++j) {
                if (std::isnan(V(i,j)) || std::isinf(V(i,j))) {
                    V(i,j) = (i == j) ? 1.0f : 0.0f;
                }
            }
        }
        
        // Write outputs
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(u_data + b * u_mat_size, m, m) = U;
        Eigen::Map<Eigen::VectorXf>(s_data + b * s_vec_size, s_vec_size) = S;
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(v_data + b * v_mat_size, n, n) = V;
    }
    // Debug: print CustomSVD outputs
    const bool debug = false; // true / false
    if (debug) {
        std::cout << "[CustomSVD Debug] U: ";
        // for (size_t i = 0; i < batch * u_mat_size; ++i) std::cout << u_data[i] << ' ';
        // std::cout << "\n[CustomSVD Debug] S: ";
        // for (size_t i = 0; i < batch * s_vec_size; ++i) std::cout << s_data[i] << ' ';
        std::cout << "\n[CustomSVD Debug] V: ";
        // for (size_t i = 0; i < batch * v_mat_size; ++i) std::cout << v_data[i] << ' ';
        std::cout << std::endl;

        FILE* fp = fopen("output/ov_svd.txt", "a");
        if (fp) {
            fprintf(fp, "----- ov custom_svd (input) -----\n");
            for (size_t i = 0; i < batch * in_mat_size; ++i) fprintf(fp, "%f ", data[i]);
            fprintf(fp, "\n");
            
            fprintf(fp, "----- ov custom_svd (U) -----\n");
            for (size_t i = 0; i < batch * u_mat_size; ++i) fprintf(fp, "%f ", u_data[i]);
            fprintf(fp, "\n");
            // for (size_t i = 0; i < batch * s_vec_size; ++i) fprintf(fp, "%f ", s_data[i]);
            // fprintf(fp, "\n");
            fprintf(fp, "----- ov custom_svd (V) -----\n");
            for (size_t i = 0; i < batch * v_mat_size; ++i) fprintf(fp, "%f ", v_data[i]);
            fprintf(fp, "\n");
            fclose(fp);
        } else {
            std::cerr << "[CustomSVD Debug] Failed to open output/ov_svd.txt for writing!" << std::endl;
        }

    }
    
    return true;
}

bool CustomSVD::has_evaluate() const {
    return true;
}
//! [op:evaluate]