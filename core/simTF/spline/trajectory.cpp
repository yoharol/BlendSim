#include <stdexcept>
#include "simTF/spline/trajectory.h"
#include "simTF/spline/node.h"

namespace aphys {

void set_keyframe_bezier_interpolation(  //
    std::vector<MatxXd>& keyframes,      //
    std::vector<int> segments,           //
    std::vector<double> time_interval,   //
    SplineTrajectory& trajectory         //
) {
  assert(time_interval.size() == segments.size());
  int n_intervals = time_interval.size();
  assert(time_interval.size() + 1 == keyframes.size());
  int n_keyframes = keyframes.size();

  int n_verts = keyframes[0].rows();
  int dim = keyframes[0].cols();

  trajectory.keyframes.clear();
  trajectory.T_between.clear();
  trajectory.tau_stamp.clear();

  double tau = 0.0;
  for (int k = 0; k < n_intervals; k++) {
    int n_segs = segments[k];
    double T = time_interval[k];
    double delta_T = T / n_segs;
    const MatxXd& begin_p = keyframes[k];
    const MatxXd& end_p = keyframes[k + 1];
    for (int j = 0; j < n_segs; j++) {
      double t = static_cast<double>(j) / n_segs;
      KeyframeNode node{MatxXd(begin_p * (1.0 - t) + end_p * t),
                        MatxXd::Zero(n_verts, dim),  //
                        Vecxb::Zero(n_verts),        //
                        Vecxb::Zero(n_verts)};
      trajectory.keyframes.push_back(node);
      trajectory.T_between.push_back(delta_T);
      trajectory.tau_stamp.push_back(tau);
      tau += delta_T;
    }
  }
  KeyframeNode node{
      keyframes[n_keyframes - 1], MatxXd::Zero(n_verts, dim),  //
      Vecxb::Zero(n_verts),                                    //
      Vecxb::Zero(n_verts)                                     //
  };
  trajectory.keyframes.push_back(node);
  trajectory.tau_stamp.push_back(tau);
  assert(trajectory.keyframes.size() == trajectory.tau_stamp.size());
  assert(trajectory.keyframes.size() == trajectory.T_between.size() + 1);
  trajectory.n_keyframes = trajectory.keyframes.size();
  trajectory.n_verts = n_verts;
}

void copy_trajectory(             //
    const SplineTrajectory& src,  //
    SplineTrajectory& dst         //
) {
  dst.n_keyframes = src.n_keyframes;
  dst.n_verts = src.n_verts;
  dst.keyframes.resize(src.keyframes.size());
  for (int i = 0; i < src.keyframes.size(); i++) {
    dst.keyframes[i].pos = src.keyframes[i].pos;
    dst.keyframes[i].vel = src.keyframes[i].vel;
    dst.keyframes[i].p_fix = src.keyframes[i].p_fix;
    dst.keyframes[i].v_fix = src.keyframes[i].v_fix;
  }
  dst.T_between.resize(src.T_between.size());
  for (int i = 0; i < src.T_between.size(); i++) {
    dst.T_between[i] = src.T_between[i];
  }
  dst.tau_stamp.resize(src.tau_stamp.size());
  for (int i = 0; i < src.tau_stamp.size(); i++) {
    dst.tau_stamp[i] = src.tau_stamp[i];
  }
}

double compute_traj_distance(const SplineTrajectory& traj1,  //
                             const SplineTrajectory& traj2) {
  double dis = 0.0;
  for (int i = 0; i < traj1.keyframes.size(); i++) {
    dis += (traj1.keyframes[i].pos - traj2.keyframes[i].pos).squaredNorm();
    if (i > 0)
      dis += (traj1.T_between[i - 1] * traj1.keyframes[i].vel -
              traj2.T_between[i - 1] * traj2.keyframes[i].vel)
                 .squaredNorm();
    if (i < traj1.keyframes.size() - 1)
      dis += (traj1.T_between[i + 1] * traj1.keyframes[i].vel -
              traj2.T_between[i + 1] * traj2.keyframes[i].vel)
                 .squaredNorm();
  }
  return dis;
}

void update_trajectory_tau(       //
    SplineTrajectory& trajectory  //
) {
  double tau = 0.0;
  trajectory.tau_stamp[0] = 0.0;
  for (int i = 0; i < trajectory.T_between.size(); i++) {
    tau += trajectory.T_between[i];
    trajectory.tau_stamp[i + 1] = tau;
  }
}

int sample_tau_on_trajectory(      //
    const double tau,              //
    SplineTrajectory& trajectory,  //
    MatxXd& pos, MatxXd& acce      //
) {
  int idx = 0;
  bool selected = false;
  for (; idx < trajectory.tau_stamp.size(); idx++) {
    if (trajectory.tau_stamp[idx] <= tau &&
        trajectory.tau_stamp[idx + 1] >= tau) {
      selected = true;
      break;
    }
  }
  assert(selected && "[simTF/spline/builder][sample_on_tau]tau out of range");
  const double T = trajectory.T_between[idx];
  double t = (tau - trajectory.tau_stamp[idx]) / T;
  const MatxXd& p1 = trajectory.keyframes[idx].pos;
  const MatxXd& p2 = trajectory.keyframes[idx + 1].pos;
  const MatxXd& v1 = trajectory.keyframes[idx].vel;
  const MatxXd& v2 = trajectory.keyframes[idx + 1].vel;
  Vec4d coeff = get_bezier_coeff(t);
  pos = coeff(0) * p1 + coeff(1) * p2 + T * coeff(2) * v1 + T * coeff(3) * v2;
  Vec4d dd_coeff = get_bezier_dd_coeff(t);
  acce = dd_coeff(0) * p1 + dd_coeff(1) * p2 + T * dd_coeff(2) * v1 +
         T * dd_coeff(3) * v2;
  return idx;
}

}  // namespace aphys
