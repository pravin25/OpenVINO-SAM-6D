// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_op.hpp"
#include <algorithm>
#include <omp.h>

using namespace TemplateExtension;

CustomAddOp::CustomAddOp(const ov::Output<ov::Node>& input, float alpha, float beta) : Op({input}), m_alpha(alpha), m_beta(beta) {
    constructor_validate_and_infer_types();
}

void CustomAddOp::validate_and_infer_types() {
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool CustomAddOp::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("alpha", m_alpha);
    visitor.on_attribute("beta", m_beta);
    return true;
}

std::shared_ptr<ov::Node> CustomAddOp::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");
    return std::make_shared<CustomAddOp>(new_args[0], m_alpha, m_beta);
}

bool CustomAddOp::has_evaluate() const {
    return true;
}

bool CustomAddOp::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto in = inputs[0];
    auto out = outputs[0];
    out.set_shape(in.get_shape());
    for (size_t i = 0; i < out.get_size(); i++) {
        out.data<float>()[i] = in.data<float>()[i] * m_alpha + m_beta;
    }
    return true;
}


// class CustomAddOp : public ov::op::Op {
// private:
//     float m_alpha;
//     float m_beta;

// public:
//     OPENVINO_OP("CustomAddOp");

//     CustomAddOp() = default;

//     CustomAddOp(const ov::Output<ov::Node>& input, float alpha, float beta) : Op({input}), m_alpha(alpha), m_beta(beta) {
//         constructor_validate_and_infer_types();
//     }

//     void validate_and_infer_types() override {
//         set_output_size(1);
//         set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
//     }

//     bool visit_attributes(ov::AttributeVisitor& visitor) override {
//         visitor.on_attribute("alpha", m_alpha);
//         visitor.on_attribute("beta", m_beta);
//         return true;
//     }

//     std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
//         OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");
//         return std::make_shared<CustomAddOp>(new_args[0], m_alpha, m_beta);
//     }

//     bool has_evaluate() const override {
//         return true;
//     }

//     bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
//         auto in = inputs[0];
//         auto out = outputs[0];
//         out.set_shape(in.get_shape());
//         for (size_t i = 0; i < out.get_size(); i++) {
//             out.data<float>()[i] = in.data<float>()[i] * m_alpha + m_beta;
//         }
//         return true;
//     }
// };