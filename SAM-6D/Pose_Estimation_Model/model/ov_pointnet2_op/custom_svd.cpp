// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_svd.hpp"
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

using namespace TemplateExtension;

//! [op:ctor]
CustomSVD::CustomSVD(const ov::Output<ov::Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CustomSVD::validate_and_infer_types() {
    // 支持任意 batch 维度，最后两维为 (M, N)
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
    // 修复：ov::Dimension 没有 min，需手动判断
    if (m.is_static() && n.is_static()) {
        s_shape.push_back(std::min(m.get_length(), n.get_length()));
    } else {
        // 动态时，保守用 m
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
    // 支持批量 SVD，输入 shape: [batch..., M, N]
    const auto& in = inputs[0];
    auto shape = in.get_shape();
    if (shape.size() < 2)
        throw std::runtime_error("CustomSVD input must have at least 2 dimensions");
    size_t rank = shape.size();
    size_t m = shape[rank - 2], n = shape[rank - 1];
    size_t batch = 1;
    for (size_t i = 0; i < rank - 2; ++i) batch *= shape[i];
    const float* data = in.data<const float>();
    // 输出 shape
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
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        // U
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(u_data + b * u_mat_size, m, m) = svd.matrixU();
        // S
        Eigen::Map<Eigen::VectorXf>(s_data + b * s_vec_size, s_vec_size) = svd.singularValues();
        // V
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(v_data + b * v_mat_size, n, n) = svd.matrixV();
    }
    return true;
}

bool CustomSVD::has_evaluate() const {
    return true;
}
//! [op:evaluate]