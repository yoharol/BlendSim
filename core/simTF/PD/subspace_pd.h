#ifndef SIMTF_PD_SUBSPACE_PD_H_
#define SIMTF_PD_SUBSPACE_PD_H_

#include <ArmorerPhys/type.h>
#include <Eigen/SparseLU>

#include "simTF/type.h"

namespace aphys {

template <int dim>
struct SubspaceProjectiveDynamicsSolver {
  typedef Eigen::Matrix<int, Eigen::Dynamic, dim + 1, Eigen::RowMajor> EleMat;
  typedef Eigen::Matrix<double, dim, dim> Mat;
  typedef Eigen::Matrix<double, dim, 1> Vec;

  int n_verts;
  int n_elements;
  std::vector<Mat> dx_ref_inv;
  MatxXd P;
  SparseMatd L;
  SparseMatd J;
  SparseMatd M_h2;
  MatxXd LHS;
  SparseMatd LHS_sparse;
  SparseMatd L_ext;
  double ratio;
  Energy_Jacobian_Func energy_jacobian_func;
  // Energy_Hessian_Func energy_hessian_func;
  Eigen::SparseLU<SparseMatd, Eigen::COLAMDOrdering<int>> sparse_solver;

  SubspaceProjectiveDynamicsSolver(const MatxXd& verts,
                                   const MatxXd& verts_ref,  //
                                   const EleMat& elements,   //
                                   const Vecxd& element_mass,
                                   const Vecxd& vert_mass,                   //
                                   const MatxXd& external_force, double dt,  //
                                   double stiffness_hydro,
                                   double stiffness_devia,  //
                                   const SparseMatd& weight_mat);
  void localStep(const MatxXd& verts, const EleMat& faces);
  void compute_jacobian(const MatxXd& v_p, const MatxXd& v_p_ref,
                        const EleMat& elements, Vecxd& jacobian);

  Vecxd solve_volume_sig(const Vecxd& sig, double& lambda);
  void globalStep(const SparseMatd& weight_mat, MatxXd& verts,
                  const MatxXd& verts_pred);
};

}  // namespace aphys

#endif  // SIMTF_PD_SUBSPACE_PD_H_
