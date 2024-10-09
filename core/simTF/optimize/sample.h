#ifndef SIMTF_OPTIMIZE_SAMPLE_H_
#define SIMTF_OPTIMIZE_SAMPLE_H_

#include <ArmorerPhys/type.h>

#include "simTF/spline.h"
#include "simTF/type.h"

namespace aphys {

void build_sample_batch(                   //
    std::vector<int>& n_samples_interval,  //
    SampleBatch& sample_batch,             //
    const SplineTrajectory& trajectory     //
);

void sample_dynamic_status(  //
    const SampleInfo& sample_info,
    const SplineTrajectory& trajectory,  //
    MatxXd& v_p, MatxXd& v_acce          //
);

}  // namespace aphys

#endif  // SIMTF_OPTIMIZE_SAMPLE_H_
