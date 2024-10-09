#include "simTF/PD/local_deform_pd.h"
#include "simTF/PD/pd.h"

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SVD>

#include <iostream>

namespace aphys {

template <int dim>
void LocalDeformSolver<dim>::initializeParams(const EleMat& elements,  //
                                              const Vecxd& element_mass,
                                              const Vecxd& vert_mass,  //
                                              const MatxXd& external_force,
                                              const double stiffness_hydro,
                                              const double stiffness_devia,
                                              const double local_stiffness) {
  params = std::make_unique<LocalDeformParams<dim>>(
      elements, element_mass, vert_mass, external_force, stiffness_hydro,
      stiffness_devia, local_stiffness);
}

template <int dim>
void LocalDeformSolver<dim>::setupDeformer(const MatxXd& verts_ref,  //
                                           const int selected_vertex) {
  if (!params) {
    std::cerr << "LocalDeformSolver: params not initialized!" << std::endl;
    return;
  }
  n_verts = verts_ref.rows();
  n_elements = params->elements.rows();
  L.resize(0, 0);
  L.resize(n_verts, n_verts);
  J.resize(0, 0);
  J.resize(n_verts, dim * n_elements);
  dx_ref_inv.resize(n_elements);
  P.resize(n_elements * dim, dim);

  MatxXd L_mat(n_verts, n_verts);
  L_mat.setZero();
  MatxXd J_mat(n_verts, dim * n_elements);
  J_mat.setZero();

  for (int j = 0; j < n_elements; j++) {
    std::vector<int> idx(dim + 1);
    for (int i = 0; i < dim + 1; i++) {
      idx[i] = params->elements(j, i);
    }

    std::vector<Vec> dx_ref(dim);
    for (int i = 0; i < dim; i++)
      dx_ref[i] =
          (verts_ref.row(idx[i + 1]) - verts_ref.row(idx[0])).transpose();
    Mat dx_ref_mat;
    for (int i = 0; i < dim; i++) {
      dx_ref_mat.col(i) = dx_ref[i];
    }
    dx_ref_inv[j] = dx_ref_mat.inverse();

    std::vector<Tripletd> tripletListGj(dim * 2);
    for (int i = 0; i < dim; i++) {
      tripletListGj[i] = Tripletd(idx[0], i, -1.0);
      tripletListGj[i + dim] = Tripletd(idx[i + 1], i, 1.0);
    }

    SparseMatd Gj(n_verts, dim);
    Gj.setFromTriplets(tripletListGj.begin(), tripletListGj.end());
    SparseMatd dx_ref_sp = Eigen::SparseView(dx_ref_inv[j]);
    Gj = Gj * dx_ref_sp;

    SparseMatd SjT(dim, dim * n_elements);
    for (int i = 0; i < dim; i++) SjT.insert(i, dim * j + i) = 1.0;

    L_mat += params->element_mass(j) *
             (params->stiffness_hydro + params->stiffness_devia) * Gj *
             Gj.transpose();
    J_mat += params->element_mass(j) *
             (params->stiffness_hydro + params->stiffness_devia) * Gj * SjT;
  }
  L = L_mat.sparseView();
  J = J_mat.sparseView();
  ratio = params->stiffness_devia /
          (params->stiffness_hydro + params->stiffness_devia);

  std::cout << "here\n";

  K.resize(0, 0);
  K.resize(n_verts, n_verts);
  for (int i = 0; i < n_verts; i++) {
    K.insert(i, i) = params->local_stiffness;
  }

  rhs_KX_ref = K * verts_ref;

  LHS_sparse = L + K;

  LHS_sparse.conservativeResize(n_verts + 1, n_verts + 1);
  LHS_sparse.insert(n_verts, selected_vertex) = 1.0;
  LHS_sparse.insert(selected_vertex, n_verts) = 1.0;

  sparse_solver.analyzePattern(LHS_sparse);
  sparse_solver.factorize(LHS_sparse);
}

template <int dim>
void LocalDeformSolver<dim>::localStep(const MatxXd& verts) {
  for (int j = 0; j < n_elements; j++) {
    Eigen::Matrix<double, dim, dim> F;
    std::vector<int> idx(dim + 1);
    std::vector<Vec> dx(dim);
    for (int i = 0; i < dim + 1; i++) idx[i] = params->elements(j, i);
    for (int i = 0; i < dim; i++)
      F.col(i) = (verts.row(idx[i + 1]) - verts.row(idx[0])).transpose();
    F = F * dx_ref_inv[j];

    Eigen::JacobiSVD<MatxXd> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    MatxXd U, V;
    Vecxd S;
    U = svd.matrixU();
    V = svd.matrixV();
    S = svd.singularValues();
    ssvd<dim>(U, S, V);

    double lambda;
    Vecxd d = solve_volume_sig<dim>(S, lambda);

    Vecxd S_new = S + d;
    Mat D = U * S_new.asDiagonal() * V.transpose();
    Mat R = U * V.transpose();
    P.block(j * dim, 0, dim, dim) = ((1.0 - ratio) * D + ratio * R).transpose();
  }
}

template <int dim>
void LocalDeformSolver<dim>::globalStep(MatxXd& verts, Vecxd fixed_pos) {
  Eigen::MatrixXd rhs = J * P + rhs_KX_ref;
  Eigen::MatrixXd rhs_expanded = Eigen::MatrixXd::Zero(n_verts + 1, dim);
  rhs_expanded.topRows(n_verts) = rhs;
  rhs_expanded.row(n_verts) = fixed_pos.transpose();
  Eigen::MatrixXd result = sparse_solver.solve(rhs_expanded);
  verts = result.topRows(n_verts);
}

template struct LocalDeformSolver<2>;
template struct LocalDeformSolver<3>;

}  // namespace aphys
