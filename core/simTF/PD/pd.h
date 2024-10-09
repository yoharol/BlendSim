#ifndef SIMTF_PD_PD_H_
#define SIMTF_PD_PD_H_

#include <ArmorerPhys/type.h>
#include <Eigen/SparseLU>

#include "simTF/type.h"

namespace aphys {

void print_tet_info(const TetMesh& tm);

// Data for projective dynamics:
//  verts: n_verts x N
//  verts_ref: n_verts x N
//  verts_cache: n_verts x N
//  elements: n_elements x (N+1)
//  element_mass: n_elements x 1
//  vert_mass: n_verts x 1
//  external_force: n_verts x N
//  dt, stiffness
// .P: 2 n_faces x N
//  L: n_verts x n_verts
//  J: n_verts x N n_faces
//  M_h2:  n_verts x n_verts (M / h^2)
// solve function: (L + M_h2) x = M_h2 * x_pred + J * P

template <int dim>
struct ProjectiveDynamicsSolver {
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
  SparseMatd J_ext;
  Vecxd exten_force_vec;
  double ratio;
  Energy_Jacobian_Func energy_jacobian_func;
  Local_Step_Func local_step_func;

  // Energy_Hessian_Func energy_hessian_func;
  Eigen::SparseLU<SparseMatd, Eigen::COLAMDOrdering<int>> sparse_solver;

  ProjectiveDynamicsSolver(const MatxXd& verts, const MatxXd& verts_ref,  //
                           const EleMat& elements,                        //
                           const Vecxd& element_mass,
                           const Vecxd& vert_mass,  //
                           const MatxXd& external_force, double dt,
                           double stiffness_hydro, double stiffness_devia);
  void localStep(const MatxXd& verts, const EleMat& elements);

  void recompute_reference(MatxXd& verts, const EleMat& elements,
                           MatxXd& new_P);

  void compute_jacobian(const MatxXd& v_p, const MatxXd& v_p_ref,
                        const EleMat& elements, Vecxd& jacobian);
};

template <int dim>
Vecxd solve_volume_sig(const Vecxd& sig, double& lambda);

}  // namespace aphys

#endif  // SIMTF_PD_PD_H_
