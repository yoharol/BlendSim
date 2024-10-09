#include "simTF/optimize/argPV.h"
#include "simTF/spline.h"

#include <ArmorerPhys/type.h>
#include <ArmorerPhys/glmath.h>
#include <iostream>

namespace aphys {

void initialize_argPV(                               //
    const SplineTrajectory& keyframe_interpolation,  //
    ArgIdxPV& arg_idx_pv                             //
) {
  arg_idx_pv.n_keyframes = keyframe_interpolation.n_keyframes;
  arg_idx_pv.n_argP = 0;
  arg_idx_pv.n_argV = 0;
  arg_idx_pv.n_fixed_argP = 0;
  arg_idx_pv.n_fixed_argV = 0;
  arg_idx_pv.keyframe_idx.clear();
  int n_verts = keyframe_interpolation.n_verts;

  for (int k = 0; k < arg_idx_pv.n_keyframes; k++) {
    const Vecxb& P_fix = keyframe_interpolation.keyframes[k].p_fix;
    const Vecxb& V_fix = keyframe_interpolation.keyframes[k].v_fix;
    assert(P_fix.size() == n_verts);
    Vecxi P_idx_in_arg = Vecxi::Zero(n_verts);
    Vecxi V_idx_in_arg = Vecxi::Zero(n_verts);
    Vecxi P_fixed_idx_in_arg = Vecxi::Zero(n_verts);
    Vecxi V_fixed_idx_in_arg = Vecxi::Zero(n_verts);
    P_idx_in_arg.fill(-1);
    V_idx_in_arg.fill(-1);
    P_fixed_idx_in_arg.fill(-1);
    V_fixed_idx_in_arg.fill(-1);
    for (int i = 0; i < n_verts; i++) {
      if (!P_fix(i)) {
        P_idx_in_arg(i) = arg_idx_pv.n_argP;
        arg_idx_pv.n_argP++;
      } else {
        P_fixed_idx_in_arg(i) = arg_idx_pv.n_fixed_argP;
        arg_idx_pv.n_fixed_argP++;
      }
      if (!V_fix(i)) {
        V_idx_in_arg(i) = arg_idx_pv.n_argV;
        arg_idx_pv.n_argV++;
      } else {
        V_fixed_idx_in_arg(i) = arg_idx_pv.n_fixed_argV;
        arg_idx_pv.n_fixed_argV++;
      }
    }
    arg_idx_pv.keyframe_idx.push_back({P_idx_in_arg, V_idx_in_arg});
    arg_idx_pv.keyframe_fixed_idx.push_back(
        {P_fixed_idx_in_arg, V_fixed_idx_in_arg});
  }

  for (int k = 0; k < arg_idx_pv.n_keyframes; k++) {
    for (int i = 0; i < n_verts; i++) {
      if (arg_idx_pv.keyframe_idx[k].V_idx_in_arg(i) != -1) {
        arg_idx_pv.keyframe_idx[k].V_idx_in_arg(i) += arg_idx_pv.n_argP;
      }
      if (arg_idx_pv.keyframe_fixed_idx[k].V_idx_in_arg(i) != -1) {
        arg_idx_pv.keyframe_fixed_idx[k].V_idx_in_arg(i) +=
            arg_idx_pv.n_fixed_argP;
      }
    }
  }
}

template <int dim>
void select_optimize_argPV(                          //
    const SplineTrajectory& keyframe_interpolation,  //
    const ArgIdxPV& arg_idx, Vecxd& curr_arg         //
) {
  assert(curr_arg.size() == (arg_idx.n_argP + arg_idx.n_argV) * dim);
  int n_verts = keyframe_interpolation.n_verts;
  for (int k = 0; k < keyframe_interpolation.n_keyframes; k++) {
    const Vecxi& P_idx_in_arg = arg_idx.keyframe_idx[k].P_idx_in_arg;
    const Vecxi& V_idx_in_arg = arg_idx.keyframe_idx[k].V_idx_in_arg;
    const MatxXd& p = keyframe_interpolation.keyframes[k].pos;
    const MatxXd& v = keyframe_interpolation.keyframes[k].vel;
    for (int i = 0; i < n_verts; i++) {
      if (P_idx_in_arg(i) != -1)
        curr_arg.segment<dim>(P_idx_in_arg(i) * dim) = p.row(i).transpose();
      if (V_idx_in_arg(i) != -1)
        curr_arg.segment<dim>(V_idx_in_arg(i) * dim) = v.row(i).transpose();
    }
  }
}

template void select_optimize_argPV<2>(const SplineTrajectory&,  //
                                       const ArgIdxPV&, Vecxd&);
template void select_optimize_argPV<3>(const SplineTrajectory&,  //
                                       const ArgIdxPV&, Vecxd&);

template <int dim>
void retrive_from_argPV(                       //
    SplineTrajectory& keyframe_interpolation,  //
    const ArgIdxPV& arg_idx,                   //
    const Vecxd& curr_arg                      //
) {
  assert(curr_arg.size() == (arg_idx.n_argP + arg_idx.n_argV) * dim);
  int n_verts = keyframe_interpolation.n_verts;
  for (int k = 0; k < keyframe_interpolation.n_keyframes; k++) {
    const Vecxi& P_idx_in_arg = arg_idx.keyframe_idx[k].P_idx_in_arg;
    const Vecxi& V_idx_in_arg = arg_idx.keyframe_idx[k].V_idx_in_arg;
    MatxXd& p = keyframe_interpolation.keyframes[k].pos;
    MatxXd& v = keyframe_interpolation.keyframes[k].vel;
    for (int i = 0; i < n_verts; i++) {
      if (P_idx_in_arg(i) != -1)
        p.row(i) = curr_arg.segment<dim>(P_idx_in_arg(i) * dim).transpose();
      if (V_idx_in_arg(i) != -1)
        v.row(i) = curr_arg.segment<dim>(V_idx_in_arg(i) * dim).transpose();
    }
  }
}

template void retrive_from_argPV<2>(SplineTrajectory&, const ArgIdxPV&,
                                    const Vecxd&);
template void retrive_from_argPV<3>(SplineTrajectory&, const ArgIdxPV&,
                                    const Vecxd&);

void shrink_vector(           //
    int n_verts,              //
    const Vecxd& vector,      //
    const ArgIdxPV& arg_idx,  //
    Vecxd& vector_argPV       //
) {
  int n_keyframes = arg_idx.n_keyframes;
  int n_args = arg_idx.n_argP + arg_idx.n_argV;
  vector_argPV.resize(n_args);
  for (int k = 0; k < n_keyframes; k++) {
    const Vecxi& P_idx_in_arg = arg_idx.keyframe_idx[k].P_idx_in_arg;
    const Vecxi& V_idx_in_arg = arg_idx.keyframe_idx[k].V_idx_in_arg;
    int base = 2 * k * n_verts;
    for (int i = 0; i < n_verts; i++) {
      if (P_idx_in_arg(i) != -1) {
        vector_argPV(P_idx_in_arg(i)) = vector(base + i);
      }
      if (V_idx_in_arg(i) != -1) {
        vector_argPV(V_idx_in_arg(i)) = vector(base + n_verts + i);
      }
    }
  }
}

void shrink_hessian(           //
    int n_verts,               //
    SparseMatd& hessian,       //
    const ArgIdxPV& arg_idx,   //
    SparseMatd& hessian_argPV  //
) {
  int n_keyframes = arg_idx.n_keyframes;
  int n_args = arg_idx.n_argP + arg_idx.n_argV;
  hessian_argPV.resize(n_args, n_args);

  for_each_nonzero(hessian, [&](SparseMatdIter& iter) {
    int i = iter.row();
    int j = iter.col();

    int i_kdx = i / (2 * n_verts);
    int j_kdx = j / (2 * n_verts);

    int idx, jdx;

    const Vecxi& P_idx_in_arg = arg_idx.keyframe_idx[i_kdx].P_idx_in_arg;
    const Vecxi& V_idx_in_arg = arg_idx.keyframe_idx[i_kdx].V_idx_in_arg;

    const Vecxi& P_jdx_in_arg = arg_idx.keyframe_idx[j_kdx].P_idx_in_arg;
    const Vecxi& V_jdx_in_arg = arg_idx.keyframe_idx[j_kdx].V_idx_in_arg;

    i = i % (2 * n_verts);

    std::string namei, namej;

    if (i >= n_verts) {
      idx = V_idx_in_arg((i - n_verts));
      namei = "V";
    } else {
      idx = P_idx_in_arg(i);
      namej = "P";
    }

    j = j % (2 * n_verts);

    if (j >= n_verts) {
      namej = "V";
      jdx = V_jdx_in_arg((j - n_verts));
    } else {
      namej = "P";
      jdx = P_jdx_in_arg(j);
    }

    if (idx != -1 && jdx != -1) {
      hessian_argPV.insert(idx, jdx) += iter.value();
    }
  });
}

}  // namespace aphys
