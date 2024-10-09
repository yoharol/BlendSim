#include "simTF/PD/pd.h"

#include <ArmorerPhys/glmath.h>
#include <ArmorerPhys/timer.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>

namespace aphys {

void print_tet_info(const TetMesh& tm) {
  std::cout << "n_verts: " << tm.verts.rows() << std::endl;
  std::cout << "n_tets: " << tm.tets.rows() << std::endl;
}

template <>
Vecxd solve_volume_sig<2>(const Vecxd& sig, double& lambda) {
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
Vecxd solve_volume_sig<3>(const Vecxd& sig, double& lambda) {
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
ProjectiveDynamicsSolver<dim>::ProjectiveDynamicsSolver(
    const MatxXd& verts, const MatxXd& verts_ref,  //
    const EleMat& elements,                        //
    const Vecxd& element_mass,
    const Vecxd& vert_mass,                   //
    const MatxXd& external_force, double dt,  //
    double stiffness_hydro, double stiffness_devia) {
  n_verts = verts.rows();
  n_elements = elements.rows();
  L.resize(n_verts, n_verts);
  L_ext.resize(n_verts * dim, n_verts * dim);
  J.resize(n_verts, dim * n_elements);
  J_ext.resize(n_verts * dim, dim * n_elements * dim);
  M_h2.resize(n_verts, n_verts);
  LHS.resize(n_verts, n_verts);
  dx_ref_inv.resize(n_elements);
  P.resize(n_elements * dim, dim);

  exten_force_vec = Vecxd::Map(external_force.data(), n_verts * dim);

  MatxXd L_mat(n_verts, n_verts);
  L_mat.setZero();
  MatxXd J_mat(n_verts, dim * n_elements);
  J_mat.setZero();

  Timer::getInstance()->start("pd_solver");
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
  Timer::getInstance()->pause("pd_solver");

  ratio = stiffness_devia / (stiffness_hydro + stiffness_devia);

  Timer::getInstance()->start("copy matrix");

  // ! a peice of sparse matrix extension code thanks to chatgpt
  typedef Eigen::Triplet<double> T;

  std::vector<T> tripletListL;
  tripletListL.reserve(L.nonZeros() * dim);

  for (int k = 0; k < L.outerSize(); ++k) {
    for (SparseMatd::InnerIterator it(L, k); it; ++it) {
      int i = it.row();
      int j = it.col();
      double value = it.value();
      for (int d = 0; d < dim; ++d) {
        int row = i * dim + d;
        int col = j * dim + d;
        tripletListL.emplace_back(row, col, value);
      }
    }
  }

  L_ext.resize(L.rows() * dim, L.cols() * dim);
  L_ext.setFromTriplets(tripletListL.begin(), tripletListL.end());

  std::vector<T> tripletListJ;
  tripletListJ.reserve(J.nonZeros() * dim);

  for (int k = 0; k < J.outerSize(); ++k) {
    for (SparseMatd::InnerIterator it(J, k); it; ++it) {
      int i = it.row();
      int j = it.col();
      double value = it.value();
      for (int d = 0; d < dim; ++d) {
        int row = i * dim + d;
        int col = j * dim + d;
        tripletListJ.emplace_back(row, col, value);
      }
    }
  }

  J_ext.resize(J.rows() * dim, J.cols() * dim);
  J_ext.setFromTriplets(tripletListJ.begin(), tripletListJ.end());
  Timer::getInstance()->pause("copy matrix");

  Timer::getInstance()->print_all();

  // for (int i = 0; i < n_verts; i++) M_h2.insert(i, i) = vert_mass(i) / dt /
  // dt; M_h2.makeCompressed();

  // LHS_sparse = L + M_h2;
  // !
  // sparse_solver.analyzePattern(L);
  // sparse_solver.factorize(L);

  energy_jacobian_func = [this, &verts_ref, &elements](const MatxXd& v_p,
                                                       Vecxd& jacobian) {
    compute_jacobian(v_p, verts_ref, elements, jacobian);
  };

  local_step_func = [this, &elements](const MatxXd& verts, Vecxd& P_vec) {
    localStep(verts, elements);
    MatxXd result = J * P;
    P_vec = Eigen::Map<Vecxd>(result.data(), result.size());
  };
}

template <int dim>
void ProjectiveDynamicsSolver<dim>::localStep(const MatxXd& verts,
                                              const EleMat& elements) {
  for (int j = 0; j < n_elements; j++) {
    Eigen::Matrix<double, dim, dim> F;
    std::vector<int> idx(dim + 1);
    std::vector<Vec> dx(dim);

    for (int i = 0; i < dim + 1; i++) idx[i] = elements(j, i);
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
/*
template <int dim>
void ProjectiveDynamicsSolver<dim>::localArapInterpolate(const MatxXd& verts1,
                                                         const MatxXd& verts2,
                                                         double t,
                                                         const EleMat& elements,
                                                         MatxXd& interp_P) {
  interp_P = P;
  Eigen::Matrix<double, dim, dim> F1;
  Eigen::Matrix<double, dim, dim> F2;
  for (int j = 0; j < n_elements; j++) {
    std::vector<int> idx(dim + 1);
    for (int i = 0; i < dim + 1; i++) idx[i] = elements(j, i);
    for (int i = 0; i < dim; i++) {
      F1.col(i) = (verts1.row(idx[i + 1]) - verts1.row(idx[0])).transpose();
      F2.col(i) = (verts2.row(idx[i + 1]) - verts2.row(idx[0])).transpose();
    }
    F1 = F1 * dx_ref_inv[j];
    F2 = F2 * dx_ref_inv[j];

    Eigen::JacobiSVD<MatxXd> svd1(F1,
                                  Eigen::ComputeFullU | Eigen::ComputeFullV);
    MatxXd U1, V1;
    Vecxd S1;
    U1 = svd1.matrixU();
    V1 = svd1.matrixV();
    S1 = svd1.singularValues();
    MatxXd R1 = U1 * V1.transpose();
    MatxXd SS1 = V1 * S1.asDiagonal() * V1.transpose();


    Vecxd S_new1 = S1 + d1;

    Eigen::JacobiSVD<MatxXd> svd2(F2,
                                  Eigen::ComputeFullU | Eigen::ComputeFullV);
    MatxXd U2, V2;
    Vecxd S2;
    U2 = svd2.matrixU();
    V2 = svd2.matrixV();
    S2 = svd2.singularValues();
    ssvd<dim>(U2, S2, V2);

    double lambda2;
    Vecxd d2 = solve_volume_sig(S2, lambda2);

    Vecxd S_new2 = S2 + d2;

    // Mat D = U * S_new.asDiagonal() * V.transpose();
    // Mat R = U * V.transpose();
    // P.block(j * dim, 0, dim, dim) = ((1.0 - ratio) * D + ratio *
    // R).transpose();
  }
}*/

template <int dim>
void ProjectiveDynamicsSolver<dim>::recompute_reference(MatxXd& verts,
                                                        const EleMat& elements,
                                                        MatxXd& new_P) {
  new_P.resize(P.rows(), P.cols());
  new_P.setZero();
  localStep(verts, elements);
  for (int i = 0; i < n_elements; i++) {
    Eigen::Matrix<double, dim, dim> F =
        P.block(i * dim, 0, dim, dim).transpose();
    Eigen::JacobiSVD<MatxXd> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    MatxXd U, V;
    U = svd.matrixU();
    V = svd.matrixV();
    MatxXd R = U * V.transpose();
    P.block(i * dim, 0, dim, dim) = R.transpose();
  }

  MatxXd verts_new = verts;

  Eigen::MatrixXd rhs = J * P;
  Eigen::MatrixXd result = sparse_solver.solve(rhs);
  Vec3d diff = (result - verts).colwise().mean();
  verts = result;
  for (int i = 0; i < n_verts; i++) verts.row(i) -= diff;

  localStep(verts, elements);
  for (int i = 0; i < n_elements; i++) {
    Eigen::Matrix<double, dim, dim> F =
        P.block(i * dim, 0, dim, dim).transpose();
    Eigen::JacobiSVD<MatxXd> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    MatxXd U, V;
    U = svd.matrixU();
    V = svd.matrixV();
    MatxXd R = U * V.transpose();
    new_P.block(i * dim, 0, dim, dim) = R.transpose();
  }
}

template <int dim>
void ProjectiveDynamicsSolver<dim>::compute_jacobian(const MatxXd& v_p,
                                                     const MatxXd& v_p_ref,
                                                     const EleMat& elements,
                                                     Vecxd& jacobian) {
  localStep(v_p, elements);
  MatxXd gradient = L * v_p - J * P;
  jacobian = Vecxd::Map(gradient.data(), n_verts * dim) - exten_force_vec;
}

template struct ProjectiveDynamicsSolver<2>;
template struct ProjectiveDynamicsSolver<3>;

}  // namespace aphys
