#ifndef SIMTF_OPTIMIZE_REGULARIZE_H_
#define SIMTF_OPTIMIZE_REGULARIZE_H_

#include <ArmorerPhys/type.h>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#include "simTF/PD/pd.h"
#include "simTF/type.h"
#include "simTF/spline.h"

namespace aphys {

template <int dim>
struct Regularizer {
  typedef Eigen::Matrix<int, Eigen::Dynamic, dim + 1, Eigen::RowMajor> EleMat;

  SparseMatd Av;
  SparseMatd Bx;
  SparseMatd Bv;
  SparseMatd Cv;
  SparseMatd Dx;
  SparseMatd A;
  SparseMatd C;
  SparseMatd D;
  Eigen::SimplicialLDLT<SparseMatd> solver;

  Regularizer(int n_verts);

  void compute_regularize_gradient(const SplineTrajectory& trajectory,  //
                                   SparseMatd& sum_HR_arg);

  void add_PD_gradient(const SplineTrajectory& trajectory,  //
                       SparseMatd& L,                       //
                       SparseMatd& sum_HR);

  void regularize(const SplineTrajectory& trajectory,  //
                  SparseMatd& sum_HR, MatxXd& rhs, const ArgIdxPV& arg_pv,
                  MatxXd& new_arg);
  void recompute_reference_deformationP(
      const SampleBatch& batch, const SplineTrajectory& trajectory,
      ProjectiveDynamicsSolver<dim>& pd_solver, const EleMat& elemnts,
      std::vector<MatxXd>& newP);
};

template <int dim>
double summarize_regularized_eularized_jacobian_PV_with_pre(
    const SampleBatch& batch,                                        //
    const ArgIdxPV& argPV,                                           //
    const SplineTrajectory& trajectory,                              //
    const Vecxd& vert_mass,                                          //
    Vecxd& sum_J_arg,                                                //
    std::vector<SparseMatd>& hessians,                               //
    std::vector<SparseMatd>& hessians_euler,                         //
    const std::vector<Energy_Jacobian_Func>& energy_jacobian_funcs,  //
    const double h,                                                  //
    const SplineTrajectory& pred_trajectory);

}  // namespace aphys

#endif  // SIMTF_OPTIMIZE_REGULARIZE_H_
