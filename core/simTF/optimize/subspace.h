#ifndef SIMTF_OPTIMIZE_SUBSPACE_H_
#define SIMTF_OPTIMIZE_SUBSPACE_H_

#include <ArmorerPhys/type.h>
#include <ArmorerPhys/sim/pd.h>

#include "simTF/spline.h"
#include "simTF/type.h"
#include "simTF/optimize/newton.h"
#include "simTF/optimize/damping_version.h"
#include "simTF/PD/pd.h"
#include "simTF/lbs/model.h"

#include <iostream>

namespace aphys {

struct LBSDataManager {
  const int n_verts;
  const int n_controls;
  SparseMatd M;
  const SparseMatd& U;
  const SparseMatd& U_ext;
  const SparseMatd& L;
  const Matx4i& tets;
  SparseMatd LU;
  SparseMatd LU_ext;
  SparseMatd MU_ext;

  LBSDataManager(const int n_controls,
                 const ProjectiveDynamicsSolver<3>& pd_solver,
                 const LBSModel& lbs_model, const TetMesh& tm,
                 const DiagMatxXd& M)
      : n_verts(tm.verts.rows()),
        n_controls(n_controls),
        M(M),
        U(lbs_model.lbs_weights),
        U_ext(lbs_model.lbs_weights_ext),
        L(pd_solver.L),
        tets(tm.tets) {
    LU = L * U;
    LU_ext = pd_solver.L_ext * U_ext;
    MU_ext = M * U_ext;
  }
};

struct LBSDataManager2D {
  const int n_verts;
  const int n_controls;
  SparseMatd M;
  const SparseMatd& U;
  const SparseMatd& U_ext;
  const SparseMatd& L;
  const Matx3i& faces;
  SparseMatd LU;
  SparseMatd LU_ext;
  SparseMatd MU_ext;

  LBSDataManager2D(const int n_controls,
                   const ProjectiveDynamicsSolver2D& pd_solver,
                   const LBSModel2D& lbs_model, const MatxXd& v_p,
                   const Matx3i& faces, const DiagMatxXd& M)
      : n_verts(v_p.rows()),
        n_controls(n_controls),
        M(M),
        U(lbs_model.lbs_weights),
        U_ext(lbs_model.lbs_weights_ext),
        L(pd_solver.L),
        faces(faces) {
    LU = L * U;
    LU_ext = pd_solver.L_ext * U_ext;
    MU_ext = M * U_ext;
  }
};

double compute_lbs_wsp(const SampleBatch& batch, const int n_verts,
                       const SplineTrajectory& lbs_trajectory,
                       ProjectiveDynamicsSolver<3>& pd_solver,
                       const LBSDataManager& ldm, const double damping_alpha);

double compute_lbs2d_wsp(const SampleBatch& batch, const int n_verts,
                         const SplineTrajectory& lbs_trajectory,
                         ProjectiveDynamicsSolver2D& pd_solver,
                         const LBSDataManager2D& ldm,
                         const double damping_alpha);

void precompute_lbs_hessian_batch(const SampleBatch& batch,
                                  const ArgIdxPV& argPV,
                                  const std::vector<double> T_between,
                                  const LBSDataManager& ldm,
                                  const double damping_alpha,
                                  std::vector<SparseMatd>& Hessian_P,
                                  SparseMatd& sum_hessian, double h);

double summarize_lbs_jacobian_PV_with_pre(
    const SampleBatch& batch,                  //
    const ArgIdxPV& argPV,                     //
    const SplineTrajectory& trajectory,        //
    ProjectiveDynamicsSolver<3>& pd_solver,    //
    const LBSDataManager& ldm,                 //
    Vecxd& sum_J_arg,                          //
    const std::vector<SparseMatd>& Hessian_p,  //
    const double damping_alpha);

}  // namespace aphys

#endif  // SIMTF_OPTIMIZE_SUBSPACE_H_
