//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClLogWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <cl/ClTensorHandle.hpp>

#include <boost/cast.hpp>

namespace armnn
{

arm_compute::Status ClLogWorkloadValidate(const TensorInfo& input, const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::CLLogLayer::validate(&aclInput, &aclOutput);
}

ClLogWorkload::ClLogWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<ElementwiseUnaryQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClLogWorkload", 1, 1);

    arm_compute::ICLTensor& input  = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_LogLayer.configure(&input, &output);
}

void ClLogWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClLogWorkload_Execute");
    RunClFunction(m_LogLayer, CHECK_LOCATION());
}

} // namespace armnn
