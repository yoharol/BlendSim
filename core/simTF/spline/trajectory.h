#ifndef SIMTF_SPLINE_TRAJECTORY_H_
#define SIMTF_SPLINE_TRAJECTORY_H_

#include <vector>

#include "simTF/spline/node.h"

namespace aphys {

struct SplineTrajectory {
  int n_keyframes;
  int n_verts;
  std::vector<KeyframeNode> keyframes;
  std::vector<double> T_between;
  std::vector<double> tau_stamp;
};

struct ArgIdxPV {
  int n_keyframes;
  int n_argP;
  int n_argV;
  std::vector<KeyframeIdxPV> keyframe_idx;
  int n_fixed_argP;
  int n_fixed_argV;
  std::vector<KeyframeIdxPV> keyframe_fixed_idx;
};

struct SampleBatch {
  int n_samples;
  std::vector<SampleInfo> samples;
};

void set_keyframe_bezier_interpolation(std::vector<MatxXd>& keyframes,  //
                                       std::vector<int> segments,
                                       std::vector<double> time_interval,  //
                                       SplineTrajectory& trajectory);

void copy_trajectory(const SplineTrajectory& src,  //
                     SplineTrajectory& dst);

double compute_traj_distance(const SplineTrajectory& traj1,  //
                             const SplineTrajectory& traj2);

int sample_tau_on_trajectory(const double tau,              //
                             SplineTrajectory& trajectory,  //
                             MatxXd& pos, MatxXd& acce);

void update_trajectory_tau(SplineTrajectory& trajectory);

}  // namespace aphys

#endif  // SIMTF_SPLINE_TRAJECTORY_H_
