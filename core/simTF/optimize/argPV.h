#ifndef SIMTF_OPTIMIZE_ARGPV_H_
#define SIMTF_OPTIMIZE_ARGPV_H_

#include <ArmorerPhys/type.h>
#include "simTF/spline.h"

namespace aphys {

void initialize_argPV(                               //
    const SplineTrajectory& keyframe_interpolation,  //
    ArgIdxPV& arg_idx_pv                             //
);

template <int dim>
void select_optimize_argPV(                          //
    const SplineTrajectory& keyframe_interpolation,  //
    const ArgIdxPV& arg_idx, Vecxd& curr_arg         //
);

template <int dim>
void retrive_from_argPV(                       //
    SplineTrajectory& keyframe_interpolation,  //
    const ArgIdxPV& arg_idx,                   //
    const Vecxd& curr_arg                      //
);

void shrink_vector(           //
    int n_verts,              //
    const Vecxd& vector,      //
    const ArgIdxPV& arg_idx,  //
    Vecxd& vector_argPV       //
);

void shrink_hessian(           //
    int n_verts,               //
    SparseMatd& hessian,       //
    const ArgIdxPV& arg_idx,   //
    SparseMatd& hessian_argPV  //
);

}  // namespace aphys

#endif  // SIMTF_OPTIMIZE_ARGPV_H_
