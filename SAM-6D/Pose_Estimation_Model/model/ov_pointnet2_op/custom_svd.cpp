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
    // Operation doesn't change shapes end element type
    // 输入shape: (m, n)
    // auto input_shape = get_input_partial_shape(0);
    // auto input_shape =  input(0).get_partial_shape();
    // int m = input_shape[0]; // batch size
    // int n = input_shape[1]; 
    
    const auto& svd_input = input(0);
    auto svd_shape = svd_input.get_partial_shape();
    auto m = svd_shape[0];
    auto n = svd_shape[1];
    auto elem_type = get_input_element_type(0);

    // U: (m, m), S: (min(m, n)), V: (n, n)
    set_output_type(0, elem_type, ov::PartialShape{m, m}); // U
    set_output_type(1, elem_type, ov::PartialShape{m, n}); // S
    set_output_type(2, elem_type, ov::PartialShape{n, n}); // V
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
    // 1. 读取输入
    const auto& in = inputs[0];
    auto shape = in.get_shape();
    if (shape.size() != 2)
        throw std::runtime_error("Input must be 2D");

    size_t m = shape[0], n = shape[1];
    const float* data = in.data<const float>();

    // 2. 构造Eigen矩阵
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(data, m, n);

    // 3. SVD分解
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // 4. 输出U
    auto& outU = outputs[0];
    outU.set_shape({m, m});
    float* u_data = outU.data<float>();
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(u_data, m, m) = svd.matrixU();

    // 5. 输出S
    auto& outS = outputs[1];
    outS.set_shape({std::min(m, n)});
    float* s_data = outS.data<float>();
    Eigen::Map<Eigen::VectorXf>(s_data, std::min(m, n)) = svd.singularValues();

    // 6. 输出V
    auto& outV = outputs[2];
    outV.set_shape({n, n});
    float* v_data = outV.data<float>();
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(v_data, n, n) = svd.matrixV();

    return true;
}

bool CustomSVD::has_evaluate() const {
    return true;
}
//! [op:evaluate]