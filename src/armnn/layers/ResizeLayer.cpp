//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ResizeLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

using namespace armnnUtils;

namespace armnn
{

ResizeLayer::ResizeLayer(const ResizeDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Resize, param, name)
{
}

std::unique_ptr<IWorkload> ResizeLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ResizeQueueDescriptor descriptor;
    return factory.CreateResize(descriptor, PrepInfoAndDesc(descriptor));
}

ResizeLayer* ResizeLayer::Clone(Graph& graph) const
{
    return CloneBase<ResizeLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ResizeLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 1);

    const TensorShape& inputShape = inputShapes[0];
    const DataLayoutIndexed dimensionIndices = m_Param.m_DataLayout;

    unsigned int outHeight = 0, outWidth = 0;
    bool outHeightSpecified = true, outWidthSpecified = true;
    if (m_Param.m_SizeMode == ResizeDescriptor::SizeMode::Size)
    {
        outHeight = static_cast<unsigned int>(m_Param.m_TargetHeight);
        outWidth = static_cast<unsigned int>(m_Param.m_TargetWidth);
    }
    else
    {
        // SizeMode::Scale. The H and W of the input tensor might be unspecified at this stage,
        // so we have to take that into account.
        if(inputShape.AreAllDimensionsSpecified())
        {
            auto baseHeight = static_cast<float>(inputShape[dimensionIndices.GetHeightIndex()]);
            auto baseWidth = static_cast<float>(inputShape[dimensionIndices.GetWidthIndex()]);
            outHeight = static_cast<unsigned int>(baseHeight * m_Param.m_TargetHeight);
            outWidth = static_cast<unsigned int>(baseWidth * m_Param.m_TargetWidth);
        }
        else
        {
            outHeightSpecified = outWidthSpecified = false;
        }
    }

    unsigned int outChannels = inputShape[dimensionIndices.GetChannelsIndex()];
    unsigned int outBatch = inputShape[0];

    TensorShape tensorShape = m_Param.m_DataLayout == armnn::DataLayout::NHWC ?
        TensorShape( { outBatch, outHeight, outWidth, outChannels }, { true, outHeightSpecified, outWidthSpecified, true }) :
        TensorShape( { outBatch, outChannels, outHeight, outWidth }, { true, true, outHeightSpecified, outWidthSpecified });

    if (m_Param.m_HalfPixelCenters && m_Param.m_AlignCorners)
    {
        throw LayerValidationException("ResizeLayer: AlignCorners cannot be true when HalfPixelCenters is true");
    }

    return std::vector<TensorShape>({ tensorShape });
}

void ResizeLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "ResizeLayer");
}

void ResizeLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitResizeLayer(this, GetParameters(), GetName());
}

} // namespace armnn
