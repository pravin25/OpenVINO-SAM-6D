// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_det.hpp"
#include <Eigen/Dense>
#include <stdexcept>

using namespace TemplateExtension;

//! [op:ctor]
CustomDet::CustomDet(const ov::Output<ov::Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CustomDet::validate_and_infer_types() {
    // 输入shape: (m, n)
    const auto& det_input = input(0);
    auto det_shape = det_input.get_partial_shape();
    auto elem_type = get_input_element_type(0);
    // 输出为单个 float
    set_output_type(0, elem_type, ov::PartialShape{}); // scalar
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> CustomDet::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<CustomDet>(new_args.at(0));
}
//! [op:copy]

//! [op:visit_attributes]
bool CustomDet::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool CustomDet::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // 1. 读取输入
    const auto& in = inputs[0];
    auto shape = in.get_shape();
    if (shape.size() != 2 || shape[0] != shape[1])
        throw std::runtime_error("Input must be a square 2D matrix");

    size_t n = shape[0];
    const float* data = in.data<const float>();

    // 2. 构造Eigen矩阵
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(data, n, n);

    // 3. 计算行列式
    float det = A.determinant();

    // 4. 输出
    auto& out = outputs[0];
    out.set_shape({}); // scalar
    float* out_data = out.data<float>();
    out_data[0] = det;

    return true;
}

bool CustomDet::has_evaluate() const {
    return true;
}
//! [op:evaluate]
