#include "simTF/optimize/sample.h"
#include "simTF/spline.h"
#include "simTF/optimize/argPV.h"

#include <stdexcept>
#include <iostream>

namespace aphys {

void build_sample_batch(                   //
    std::vector<int>& n_samples_interval,  //
    SampleBatch& sample_batch,             //
    const SplineTrajectory& trajectory     //
) {
  assert(n_samples_interval.size() == trajectory.n_keyframes - 1);
  sample_batch.n_samples = 0;
  sample_batch.samples.clear();
  for (int s = 0; s < n_samples_interval.size(); s++) {
    int n_samples = n_samples_interval[s];
    sample_batch.n_samples += n_samples;
    double begin_tau = trajectory.tau_stamp[s];
    double weight = 1.0 / n_samples;
    for (int i = 0; i < n_samples; i++) {
      double tau = begin_tau + i * trajectory.T_between[s] / n_samples;
      double tau_next = tau + trajectory.T_between[s] / n_samples;
      if (s == n_samples_interval.size() - 1 && i == n_samples - 1) {
        tau = trajectory.tau_stamp[s + 1];
        tau_next = tau;
      }
      double t = (tau - begin_tau) / trajectory.T_between[s];
      SampleInfo sample{tau, tau_next, weight, t, s};
      sample_batch.samples.push_back(sample);
    }
  }
}

void sample_dynamic_status(  //
    const SampleInfo& sample_info,
    const SplineTrajectory& trajectory,  //
    MatxXd& v_p, MatxXd& v_acce          //
) {
  const int idx = sample_info.keyframe_idx;
  const double t = sample_info.t;
  Vec4d coeff = get_bezier_coeff(t);
  Vec4d dd_coeff = get_bezier_dd_coeff(t);
  const MatxXd& p1 = trajectory.keyframes[idx].pos;
  const MatxXd& p2 = trajectory.keyframes[idx + 1].pos;
  const MatxXd& v1 = trajectory.keyframes[idx].vel;
  const MatxXd& v2 = trajectory.keyframes[idx + 1].vel;
  const double T = trajectory.T_between[idx];

  v_p = coeff(0) * p1 + coeff(1) * p2 + T * coeff(2) * v1 + T * coeff(3) * v2;
  v_acce = dd_coeff(0) * p1 + dd_coeff(1) * p2 + T * dd_coeff(2) * v1 +
           T * dd_coeff(3) * v2;
}

}  // namespace aphys
