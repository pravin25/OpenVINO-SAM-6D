#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "ball_query.hpp"
#include "custom_op.hpp"

// clang-format off
//! [ov_extension:entry_point]
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<TemplateExtension::BallQuery>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::BallQuery>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomAddOp>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomAddOp>>(),
    }));
//! [ov_extension:entry_point]
// clang-format on