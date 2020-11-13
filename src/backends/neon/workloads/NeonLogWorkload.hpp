//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <arm_compute/core/Error.h>
#include <arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h>

namespace armnn
{

arm_compute::Status NeonLogWorkloadValidate(const TensorInfo& input, const TensorInfo& output);

class NeonLogWorkload : public BaseWorkload<ElementwiseUnaryQueueDescriptor>
{
public:
    NeonLogWorkload(const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info);
    virtual void Execute() const override;

private:
    mutable arm_compute::NELogLayer m_LogLayer;
};

} //namespace armnn





