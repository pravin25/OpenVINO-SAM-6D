#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "custom_op.hpp"

#include "ball_query.hpp"
#include "custom_det.hpp"

#include "custom_svd.hpp"
#include "custom_svd_u.hpp"
#include "custom_svd_v.hpp"

#include "furthest_point_sampling.hpp"
#include "gather_operation.hpp"
#include "grouping_operation.hpp"

// clang-format off
//! [ov_extension:entry_point]
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<TemplateExtension::CustomAddOp>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomAddOp>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::BallQuery>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::BallQuery>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomDet>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomDet>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomSVD>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomSVD>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomSVDu>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomSVDu>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomSVDv>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomSVDv>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::FurthestPointSampling>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::FurthestPointSampling>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::GatherOperation>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::GatherOperation>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::GroupingOperation>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::GroupingOperation>>(),

    }));
//! [ov_extension:entry_point]
// clang-format on