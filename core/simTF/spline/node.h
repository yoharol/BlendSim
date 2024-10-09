#ifndef SIMTF_SPLINE_NODE_H
#define SIMTF_SPLINE_NODE_H

#include "ArmorerPhys/type.h"

namespace aphys {

struct KeyframeNode {
  MatxXd pos;
  MatxXd vel;
  Vecxb p_fix;
  Vecxb v_fix;
};

struct KeyframeIdxPV {
  Vecxi P_idx_in_arg;
  Vecxi V_idx_in_arg;
};

struct SampleInfo {
  double tau_stamp;
  double tau_next;
  double weight;
  double t;
  int keyframe_idx;
};

Vec4d get_bezier_coeff(const double t);
Vec4d get_bezier_d_coeff(const double t);
Vec4d get_bezier_dd_coeff(const double t);

}  // namespace aphys

#endif  // STSP_SPLINE_NODE_H
