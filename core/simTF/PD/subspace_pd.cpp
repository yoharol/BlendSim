#include "simTF/PD/subspace_pd.h"

#include <ArmorerPhys/glmath.h>
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace aphys {

template <>
Vecxd SubspaceProjectiveDynamicsSolver<2>::solve_volume_sig(const Vecxd& sig,
                                                            double& lambda) {
  double sig11 = sig(0);
  double sig22 = sig(1);
  double d1, d2;
  d1 = 1.0 - sig11;
  d2 = 1.0 - sig22;
  lambda = 0.0f;
  for (int iter = 0; iter < 20; iter++) {
    MatxXd lhs(3, 3);
    lhs << 1.0f, lambda, d2 + sig22,   //
        lambda, 1.0f, d1 + sig11,      //
        d2 + sig22, d1 + sig11, 0.0f;  //
    Vecxd rhs(3);
    rhs << d1 + lambda * (d2 + sig22), d2 + lambda * (d1 + sig11),
        (d1 + sig11) * (d2 + sig22) - 1.0f;
    rhs = -rhs;
    Vecxd delta = lhs.partialPivLu().solve(rhs);
    if ((delta(0) * delta(0) + delta(1) * delta(1)) < 1e-6) break;
    d1 += delta(0);
    d2 += delta(1);
    lambda += delta(2);
  }
  Vecxd d(2);
  d << d1, d2;
  return d;
}

template <>
Vecxd SubspaceProjectiveDynamicsSolver<3>::solve_volume_sig(const Vecxd& sig,
                                                            double& lambda) {
  Vecxd d(3);
  d << 1.0 - sig(0), 1.0 - sig(1), 1.0 - sig(2);
  lambda = 0.0;
  MatxXd lhs(4, 4);
  Vecxd rhs(4);
  for (int iter = 0; iter < 20; iter++) {
    double p1 = d(0) + sig(0);
    double p2 = d(1) + sig(1);
    double p3 = d(2) + sig(2);

    lhs << 1.0, lambda * p3, lambda * p2, p2 * p3,  //
        lambda * p3, 1.0, lambda * p1, p1 * p3,     //
        lambda * p2, lambda * p1, 1.0, p1 * p2,     //
        p2 * p3, p1 * p3, p1 * p2, 0.0;             //
    rhs << d(0) + lambda * p2 * p3,                 //
        d(1) + lambda * p1 * p3,                    //
        d(2) + lambda * p1 * p2,                    //
        p1 * p2 * p3 - 1.0;
    rhs = -rhs;
    Vecxd delta = lhs.partialPivLu().solve(rhs);
    if ((delta.squaredNorm()) < 1e-6) break;
    d(0) += delta(0);
    d(1) += delta(1);
    d(2) += delta(2);
    lambda += delta(3);
  }
  return d;
}

template <int dim>
SubspaceProjectiveDynamicsSolver<dim>::SubspaceProjectiveDynamicsSolver(
    const MatxXd& verts, const MatxXd& verts_ref,  //
    const EleMat& elements,                        //
    const Vecxd& element_mass,
    const Vecxd& vert_mass,                          //
    const MatxXd& external_force, double dt,         //
    double stiffness_hydro, double stiffness_devia,  //
    const SparseMatd& weight_mat) {
  n_verts = verts.rows();
  n_elements = elements.rows();
  L.resize(n_verts, n_verts);
  J.resize(n_verts, dim * n_elements);
  L_ext.resize(n_verts * dim, n_verts * dim);
  M_h2.resize(n_verts, n_verts);
  LHS.resize(n_verts, n_verts);
  dx_ref_inv.resize(n_elements);
  P.resize(n_elements * dim, dim);

  MatxXd L_mat(n_verts, n_verts);
  L_mat.setZero();
  MatxXd J_mat(n_verts, dim * n_elements);
  J_mat.setZero();

  for (int j = 0; j < n_elements; j++) {
    std::vector<int> idx(dim + 1);
    for (int i = 0; i < dim + 1; i++) {
      idx[i] = elements(j, i);
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

    L_mat += element_mass(j) * (stiffness_hydro + stiffness_devia) * Gj *
             Gj.transpose();
    J_mat += element_mass(j) * (stiffness_hydro + stiffness_devia) * Gj * SjT;
  }
  L = L_mat.sparseView();
  J = J_mat.sparseView();
  ratio = stiffness_devia / (stiffness_hydro + stiffness_devia);

  for_each_nonzero(L, [&](const SparseMatd::InnerIterator& it) {
    for (int d = 0; d < dim; d++)
      L_ext.insert(it.row() * dim + d, it.col() * dim + d) = it.value();
  });

  for (int i = 0; i < n_verts; i++) M_h2.insert(i, i) = vert_mass(i) / dt / dt;
  M_h2.makeCompressed();

  LHS_sparse = weight_mat.transpose() * (L + M_h2) * weight_mat;
  sparse_solver.analyzePattern(LHS_sparse);
  sparse_solver.factorize(LHS_sparse);

  energy_jacobian_func = [this, &verts_ref, &elements](const MatxXd& v_p,
                                                       Vecxd& jacobian) {
    compute_jacobian(v_p, verts_ref, elements, jacobian);
  };
}

template <int dim>
void SubspaceProjectiveDynamicsSolver<dim>::localStep(const MatxXd& verts,
                                                      const EleMat& elements) {
  MatxXd U, V;
  Vecxd S;
  Eigen::Matrix<double, dim, dim> F;
  for (int j = 0; j < n_elements; j++) {
    std::vector<int> idx(dim + 1);
    std::vector<Vec> dx(dim);
    for (int i = 0; i < dim + 1; i++) idx[i] = elements(j, i);
    for (int i = 0; i < dim; i++)
      F.col(i) = (verts.row(idx[i + 1]) - verts.row(idx[0])).transpose();
    F = F * dx_ref_inv[j];

    Eigen::JacobiSVD<MatxXd> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    V = svd.matrixV();
    S = svd.singularValues();
    ssvd<dim>(U, S, V);

    double lambda;
    Vecxd d = solve_volume_sig(S, lambda);

    Vecxd S_new = S + d;
    Mat D = U * S_new.asDiagonal() * V.transpose();
    Mat R = U * V.transpose();
    P.block(j * dim, 0, dim, dim) = ((1.0 - ratio) * D + ratio * R).transpose();
  }
}

template <int dim>
void SubspaceProjectiveDynamicsSolver<dim>::compute_jacobian(
    const MatxXd& v_p, const MatxXd& v_p_ref, const EleMat& elements,
    Vecxd& jacobian) {
  localStep(v_p, elements);
  MatxXd gradient = L * v_p - J * P;
  jacobian = Vecxd::Map(gradient.data(), n_verts * dim);
}

template <int dim>
void SubspaceProjectiveDynamicsSolver<dim>::globalStep(
    const SparseMatd& weight_mat, MatxXd& verts, const MatxXd& verts_pred) {
  Eigen::MatrixXd rhs = weight_mat.transpose() * (M_h2 * verts_pred + J * P);
  Eigen::MatrixXd result = sparse_solver.solve(rhs);
  verts = result;
}

template struct SubspaceProjectiveDynamicsSolver<2>;
template struct SubspaceProjectiveDynamicsSolver<3>;

}  // namespace aphys
