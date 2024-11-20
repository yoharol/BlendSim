#include "simTF/optimize/full_optimize.h"

#include <iostream>
#include <ArmorerPhys/glmath.h>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/Dense>

#include "simTF/PD/pd.h"
#include "simTF/type.h"
#include "simTF/spline.h"
#include "simTF/optimize/argPV.h"

namespace aphys {

template <int dim>
DataManager<dim>::DataManager(Vecxd& vert_mass, double damping_alpha,
                              EleMat& elements, Vecxd ext_force)
    : vert_mass(vert_mass), damping_alpha(damping_alpha), elements(elements) {
  int n_verts = vert_mass.size();
  M.resize(n_verts, n_verts);
  M_ext.resize(n_verts * dim, n_verts * dim);

  std::vector<Tripletd> M_trips;
  std::vector<Tripletd> M_ext_trips;
  ext_F.resize(n_verts, dim);

  for (int i = 0; i < n_verts; i++) {
    M_trips.emplace_back(i, i, vert_mass(i));
    for (int j = 0; j < dim; j++) {
      M_ext_trips.emplace_back(i * dim + j, i * dim + j, vert_mass(i));
    }
    ext_F.row(i) = ext_force.transpose();
  }

  M.setFromTriplets(M_trips.begin(), M_trips.end());
  M_ext.setFromTriplets(M_ext_trips.begin(), M_ext_trips.end());
}

template struct DataManager<2>;
template struct DataManager<3>;

template <int dim>
ArgSelection<dim>::ArgSelection(SplineTrajectory& trajectory, ArgIdxPV& arg_pv)
    : trajectory(trajectory), arg_pv(arg_pv) {
  const int n_verts = trajectory.n_verts;
  const int n_keyframes = trajectory.n_keyframes;
  const int n_args = arg_pv.n_argP + arg_pv.n_argV;
  const int n_fullargs = 2 * n_keyframes * n_verts;

  std::vector<Tripletd> s_trips;
  S.resize(n_fullargs, n_args);
  C.resize(n_fullargs, dim);
  C.setZero();

  for (int k = 0; k < n_keyframes; k++) {
    for (int i = 0; i < n_verts; i++) {
      int idx = arg_pv.keyframe_idx[k].P_idx_in_arg(i);
      if (!trajectory.keyframes[k].p_fix(i)) {
        s_trips.emplace_back(k * n_verts * 2 + i, idx, 1.0);
      } else {
        C.row(k * n_verts * 2 + i) = trajectory.keyframes[k].pos.row(i);
      }

      idx = arg_pv.keyframe_idx[k].V_idx_in_arg(i);
      if (!trajectory.keyframes[k].v_fix(i)) {
        s_trips.emplace_back(k * n_verts * 2 + i + n_verts, idx, 1.0);
      } else {
        C.row(k * n_verts * 2 + i + n_verts) =
            trajectory.keyframes[k].vel.row(i);
      }
    }
  }

  S.setFromTriplets(s_trips.begin(), s_trips.end());
}

template <int dim>
void ArgSelection<dim>::update_selection() {
  const int n_verts = trajectory.n_verts;
  const int n_keyframes = trajectory.n_keyframes;
  const int n_args = arg_pv.n_argP + arg_pv.n_argV;
  const int n_fullargs = 2 * n_keyframes * n_verts;

  C.setZero();

  for (int k = 0; k < n_keyframes; k++) {
    for (int i = 0; i < n_verts; i++) {
      int idx = arg_pv.keyframe_idx[k].P_idx_in_arg(i);
      if (trajectory.keyframes[k].p_fix(i)) {
        C.row(k * n_verts * 2 + i) = trajectory.keyframes[k].pos.row(i);
      }

      idx = arg_pv.keyframe_idx[k].V_idx_in_arg(i);
      if (trajectory.keyframes[k].v_fix(i)) {
        C.row(k * n_verts * 2 + i + n_verts) =
            trajectory.keyframes[k].vel.row(i);
      }
    }
  }
}

template struct ArgSelection<2>;
template struct ArgSelection<3>;

template <int dim>
void BezierMatrix<dim>::ReCompute() {
  n_verts = trajectory.n_verts;
  n_keyframes = trajectory.n_keyframes;
  n_fullargs = 2 * n_keyframes * n_verts;
  n_args = arg_pv.n_argP + arg_pv.n_argV;

  sum_lhs_hessian.resize(0, 0);
  sum_lhs_hessian.resize(n_args, n_args);
  sum_rhs_hessian.resize(0, 0);
  sum_rhs_hessian.resize(n_args, n_fullargs);
  JacobianFres.clear();
  Jacobian2.clear();

  for (int s = 0; s < sample_batch.n_samples; s++) {
    const SampleInfo& info = sample_batch.samples[s];

    SparseMatd B(n_verts, n_fullargs);
    SparseMatd Bd(n_verts, n_fullargs);
    SparseMatd Bdd(n_verts, n_fullargs);

    std::vector<Tripletd> B_trips;
    std::vector<Tripletd> Bd_trips;
    std::vector<Tripletd> Bdd_trips;

    B_trips.reserve(n_verts * 4);
    Bd_trips.reserve(n_verts * 4);
    Bdd_trips.reserve(n_verts * 4);

    int idx = info.keyframe_idx;
    double T = trajectory.T_between[idx];
    double t = info.t;
    Vec4d coeff = get_bezier_coeff(t);
    Vec4d d_coeff = get_bezier_d_coeff(t);
    Vec4d dd_coeff = get_bezier_dd_coeff(t);

    for (int i = 0; i < n_verts; i++) {
      int p1_idx = idx * 2 * n_verts + i;
      int p2_idx = (idx + 1) * 2 * n_verts + i;
      int v1_idx = idx * 2 * n_verts + i + n_verts;
      int v2_idx = (idx + 1) * 2 * n_verts + i + n_verts;
      double m = data.vert_mass(i);

      B_trips.emplace_back(i, p1_idx, coeff(0));
      Bd_trips.emplace_back(i, p1_idx, m * d_coeff(0));
      Bdd_trips.emplace_back(i, p1_idx, m * dd_coeff(0));

      B_trips.emplace_back(i, p2_idx, coeff(1));
      Bd_trips.emplace_back(i, p2_idx, m * d_coeff(1));
      Bdd_trips.emplace_back(i, p2_idx, m * dd_coeff(1));

      B_trips.emplace_back(i, v1_idx, T * coeff(2));
      Bd_trips.emplace_back(i, v1_idx, m * T * d_coeff(2));
      Bdd_trips.emplace_back(i, v1_idx, m * T * dd_coeff(2));

      B_trips.emplace_back(i, v2_idx, T * coeff(3));
      Bd_trips.emplace_back(i, v2_idx, m * T * d_coeff(3));
      Bdd_trips.emplace_back(i, v2_idx, m * T * dd_coeff(3));
    }
    B.setFromTriplets(B_trips.begin(), B_trips.end());
    Bd.setFromTriplets(Bd_trips.begin(), Bd_trips.end());
    Bdd.setFromTriplets(Bdd_trips.begin(), Bdd_trips.end());

    double dtau = info.weight * T;
    SparseMatd Jq =
        1.0 / T / T * Bdd + data.damping_alpha / T * Bd + pd_solver.L * B;
    SparseMatd JqS = Jq * arg_selection.S;

    JacobianFres.push_back(JqS);
    Jacobian2.push_back(Jq);

    sum_lhs_hessian +=
        dtau * JacobianFres.back().transpose() * JacobianFres.back();
    sum_rhs_hessian +=
        dtau * JacobianFres.back().transpose() * Jacobian2.back();
  }
  solver.analyzePattern(sum_lhs_hessian);
  solver.factorize(sum_lhs_hessian);
  rhs.resize(n_args, dim);
}

template <int dim>
BezierMatrix<dim>::BezierMatrix(SplineTrajectory& trajectory, ArgIdxPV& arg_pv,
                                SampleBatch& sample_batch,
                                ProjectiveDynamicsSolver<dim>& pd_solver,
                                ArgSelection<dim>& arg_selection,
                                DataManager<dim>& data)
    : trajectory(trajectory),
      arg_pv(arg_pv),
      sample_batch(sample_batch),
      pd_solver(pd_solver),
      data(data),
      arg_selection(arg_selection) {
  ReCompute();
}

template <int dim>
void BezierMatrix<dim>::local_step() {
  rhs.setZero();

  for (int s = 0; s < sample_batch.n_samples; s++) {
    const SampleInfo& info = sample_batch.samples[s];
    int idx = info.keyframe_idx;
    double t = info.t;
    Vec4d coeff = get_bezier_coeff(t);
    Vec4d d_coeff = get_bezier_d_coeff(t);
    Vec4d dd_coeff = get_bezier_dd_coeff(t);
    const MatxXd& p1 = trajectory.keyframes[idx].pos;
    const MatxXd& p2 = trajectory.keyframes[idx + 1].pos;
    const MatxXd& v1 = trajectory.keyframes[idx].vel;
    const MatxXd& v2 = trajectory.keyframes[idx + 1].vel;
    double T = trajectory.T_between[idx];
    double dtau = info.weight * T;
    MatxXd v_p =
        coeff(0) * p1 + coeff(1) * p2 + T * coeff(2) * v1 + T * coeff(3) * v2;
    pd_solver.localStep(v_p, data.elements);
    MatxXd JP = pd_solver.J * pd_solver.P;
    rhs += dtau * JacobianFres[s].transpose() * JP;
  }
  rhs = rhs - sum_rhs_hessian * arg_selection.C;
}

template <int dim>
void BezierMatrix<dim>::global_step() {
  Eigen::MatrixXd new_arg = solver.solve(rhs);

  MatxXd new_full_arg = arg_selection.C + arg_selection.S * new_arg;
  for (int i = 0; i < n_keyframes; i++) {
    trajectory.keyframes[i].pos =
        new_full_arg.block(i * n_verts * 2, 0, n_verts, dim);
    trajectory.keyframes[i].vel =
        new_full_arg.block(i * n_verts * 2 + n_verts, 0, n_verts, dim);
  }
  std::cout << "update complete" << std::endl;
}

template struct BezierMatrix<2>;
template struct BezierMatrix<3>;

template <int dim>
BezierLBSMatrix<dim>::BezierLBSMatrix(
    SplineTrajectory& trajectory, ArgIdxPV& arg_pv, SampleBatch& sample_batch,
    ProjectiveDynamicsSolver<dim>& pd_solver, ArgSelection<dim>& arg_selection,
    DataManager<dim>& data, LBSModel& lbs_model)
    : trajectory(trajectory),
      arg_pv(arg_pv),
      sample_batch(sample_batch),
      pd_solver(pd_solver),
      data(data),
      arg_selection(arg_selection),
      lbs_model(lbs_model) {
  n_verts = trajectory.n_verts;
  n_controls = trajectory.keyframes[0].pos.rows();
  n_keyframes = trajectory.n_keyframes;
  n_fullargs = 2 * n_keyframes * n_controls;
  n_args = arg_pv.n_argP + arg_pv.n_argV;

  sum_lhs_hessian.resize(0, 0);
  sum_lhs_hessian.resize(n_args, n_args);
  sum_rhs_hessian.resize(0, 0);
  sum_rhs_hessian.resize(n_args, n_fullargs);
  JacobianFres.clear();

  std::cout << "test\n";

  for (int s = 0; s < sample_batch.n_samples; s++) {
    const SampleInfo& info = sample_batch.samples[s];

    SparseMatd B(n_controls, n_fullargs);
    SparseMatd Bd(n_controls, n_fullargs);
    SparseMatd Bdd(n_controls, n_fullargs);

    std::vector<Tripletd> B_trips;
    std::vector<Tripletd> Bd_trips;
    std::vector<Tripletd> Bdd_trips;

    B_trips.reserve(n_controls * 4);
    Bd_trips.reserve(n_controls * 4);
    Bdd_trips.reserve(n_controls * 4);

    int idx = info.keyframe_idx;
    std::cout << idx << " " << trajectory.n_keyframes << std::endl;
    double T = trajectory.T_between[idx];
    double t = info.t;
    Vec4d coeff = get_bezier_coeff(t);
    Vec4d d_coeff = get_bezier_d_coeff(t);
    Vec4d dd_coeff = get_bezier_dd_coeff(t);

    for (int i = 0; i < n_controls; i++) {
      int p1_idx = idx * 2 * n_controls + i;
      int p2_idx = (idx + 1) * 2 * n_controls + i;
      int v1_idx = idx * 2 * n_controls + i + n_controls;
      int v2_idx = (idx + 1) * 2 * n_controls + i + n_controls;

      B_trips.emplace_back(i, p1_idx, coeff(0));
      Bd_trips.emplace_back(i, p1_idx, d_coeff(0));
      Bdd_trips.emplace_back(i, p1_idx, dd_coeff(0));

      B_trips.emplace_back(i, p2_idx, coeff(1));
      Bd_trips.emplace_back(i, p2_idx, d_coeff(1));
      Bdd_trips.emplace_back(i, p2_idx, dd_coeff(1));

      B_trips.emplace_back(i, v1_idx, T * coeff(2));
      Bd_trips.emplace_back(i, v1_idx, T * d_coeff(2));
      Bdd_trips.emplace_back(i, v1_idx, T * dd_coeff(2));

      B_trips.emplace_back(i, v2_idx, T * coeff(3));
      Bd_trips.emplace_back(i, v2_idx, T * d_coeff(3));
      Bdd_trips.emplace_back(i, v2_idx, T * dd_coeff(3));
    }
    B.setFromTriplets(B_trips.begin(), B_trips.end());
    Bd.setFromTriplets(Bd_trips.begin(), Bd_trips.end());
    Bdd.setFromTriplets(Bdd_trips.begin(), Bdd_trips.end());

    double dtau = info.weight * T;
    SparseMatd& U = lbs_model.lbs_weights;
    SparseMatd& M = data.M;
    SparseMatd MU = M * U;
    SparseMatd Jq = 1.0 / T / T * MU * Bdd + data.damping_alpha / T * MU * Bd +
                    pd_solver.L * U * B;
    SparseMatd JqS = Jq * arg_selection.S;

    JacobianFres.push_back(JqS);
    Jacobian2.push_back(Jq);

    sum_lhs_hessian +=
        dtau * JacobianFres.back().transpose() * JacobianFres.back();
    sum_rhs_hessian +=
        dtau * JacobianFres.back().transpose() * Jacobian2.back();
  }
  // std::cout << sum_lhs_hessian << std::endl;
  solver.analyzePattern(sum_lhs_hessian);
  solver.factorize(sum_lhs_hessian);
  rhs.resize(n_args, dim);
}

template <int dim>
void BezierLBSMatrix<dim>::local_step() {
  rhs.setZero();

  for (int s = 0; s < sample_batch.n_samples; s++) {
    const SampleInfo& info = sample_batch.samples[s];
    int idx = info.keyframe_idx;
    double t = info.t;
    Vec4d coeff = get_bezier_coeff(t);
    Vec4d d_coeff = get_bezier_d_coeff(t);
    Vec4d dd_coeff = get_bezier_dd_coeff(t);
    const MatxXd& p1 = trajectory.keyframes[idx].pos;
    const MatxXd& p2 = trajectory.keyframes[idx + 1].pos;
    const MatxXd& v1 = trajectory.keyframes[idx].vel;
    const MatxXd& v2 = trajectory.keyframes[idx + 1].vel;
    double T = trajectory.T_between[idx];
    double dtau = info.weight * T;
    MatxXd v_p =
        coeff(0) * p1 + coeff(1) * p2 + T * coeff(2) * v1 + T * coeff(3) * v2;
    MatxXd x = lbs_model.lbs_weights * v_p;
    pd_solver.localStep(x, data.elements);
    MatxXd JP = pd_solver.J * pd_solver.P;
    rhs += dtau * JacobianFres[s].transpose() * JP;
  }
  rhs = rhs - sum_rhs_hessian * arg_selection.C;
}

template <int dim>
void BezierLBSMatrix<dim>::global_step() {
  Eigen::MatrixXd new_arg = solver.solve(rhs);

  MatxXd new_full_arg = arg_selection.C + arg_selection.S * new_arg;
  for (int i = 0; i < n_keyframes; i++) {
    trajectory.keyframes[i].pos =
        new_full_arg.block(i * n_controls * 2, 0, n_controls, dim);
    trajectory.keyframes[i].vel =
        new_full_arg.block(i * n_controls * 2 + n_controls, 0, n_controls, dim);
  }
  std::cout << "update complete" << std::endl;
}

template struct BezierLBSMatrix<2>;
template struct BezierLBSMatrix<3>;

BezierLBS2D::BezierLBS2D(SplineTrajectory& trajectory, ArgIdxPV& arg_pv,
                         SampleBatch& sample_batch,
                         ProjectiveDynamicsSolver2D& pd_solver,
                         ArgSelection<2>& arg_selection, DataManager<2>& data,
                         LBSModel2D& lbs_model)
    : trajectory(trajectory),
      arg_pv(arg_pv),
      sample_batch(sample_batch),
      pd_solver(pd_solver),
      data(data),
      arg_selection(arg_selection),
      lbs_model(lbs_model) {
  n_verts = trajectory.n_verts;
  n_controls = trajectory.keyframes[0].pos.rows();
  n_keyframes = trajectory.n_keyframes;
  n_fullargs = 2 * n_keyframes * n_controls;
  n_args = arg_pv.n_argP + arg_pv.n_argV;

  sum_lhs_hessian.resize(0, 0);
  sum_lhs_hessian.resize(n_args, n_args);
  sum_rhs_hessian.resize(0, 0);
  sum_rhs_hessian.resize(n_args, n_fullargs);
  JacobianFres.clear();

  for (int s = 0; s < sample_batch.n_samples; s++) {
    const SampleInfo& info = sample_batch.samples[s];

    SparseMatd B(n_controls, n_fullargs);
    SparseMatd Bd(n_controls, n_fullargs);
    SparseMatd Bdd(n_controls, n_fullargs);

    std::vector<Tripletd> B_trips;
    std::vector<Tripletd> Bd_trips;
    std::vector<Tripletd> Bdd_trips;

    B_trips.reserve(n_controls * 4);
    Bd_trips.reserve(n_controls * 4);
    Bdd_trips.reserve(n_controls * 4);

    int idx = info.keyframe_idx;
    double T = trajectory.T_between[idx];
    double t = info.t;
    Vec4d coeff = get_bezier_coeff(t);
    Vec4d d_coeff = get_bezier_d_coeff(t);
    Vec4d dd_coeff = get_bezier_dd_coeff(t);

    for (int i = 0; i < n_controls; i++) {
      int p1_idx = idx * 2 * n_controls + i;
      int p2_idx = (idx + 1) * 2 * n_controls + i;
      int v1_idx = idx * 2 * n_controls + i + n_controls;
      int v2_idx = (idx + 1) * 2 * n_controls + i + n_controls;

      B_trips.emplace_back(i, p1_idx, coeff(0));
      Bd_trips.emplace_back(i, p1_idx, d_coeff(0));
      Bdd_trips.emplace_back(i, p1_idx, dd_coeff(0));

      B_trips.emplace_back(i, p2_idx, coeff(1));
      Bd_trips.emplace_back(i, p2_idx, d_coeff(1));
      Bdd_trips.emplace_back(i, p2_idx, dd_coeff(1));

      B_trips.emplace_back(i, v1_idx, T * coeff(2));
      Bd_trips.emplace_back(i, v1_idx, T * d_coeff(2));
      Bdd_trips.emplace_back(i, v1_idx, T * dd_coeff(2));

      B_trips.emplace_back(i, v2_idx, T * coeff(3));
      Bd_trips.emplace_back(i, v2_idx, T * d_coeff(3));
      Bdd_trips.emplace_back(i, v2_idx, T * dd_coeff(3));
    }
    B.setFromTriplets(B_trips.begin(), B_trips.end());
    Bd.setFromTriplets(Bd_trips.begin(), Bd_trips.end());
    Bdd.setFromTriplets(Bdd_trips.begin(), Bdd_trips.end());

    double dtau = info.weight * T;
    SparseMatd& U = lbs_model.lbs_weights;
    SparseMatd& M = data.M;
    SparseMatd MU = M * U;
    SparseMatd Jq = 1.0 / T / T * MU * Bdd + data.damping_alpha / T * MU * Bd +
                    pd_solver.L * U * B;
    SparseMatd JqS = Jq * arg_selection.S;

    JacobianFres.push_back(JqS);
    Jacobian2.push_back(Jq);

    sum_lhs_hessian +=
        dtau * JacobianFres.back().transpose() * JacobianFres.back();
    sum_rhs_hessian +=
        dtau * JacobianFres.back().transpose() * Jacobian2.back();
  }
  solver.analyzePattern(sum_lhs_hessian);
  solver.factorize(sum_lhs_hessian);
  rhs.resize(n_args, 2);
}

void BezierLBS2D::local_step() {
  rhs.setZero();

  for (int s = 0; s < sample_batch.n_samples; s++) {
    const SampleInfo& info = sample_batch.samples[s];
    int idx = info.keyframe_idx;
    double t = info.t;
    Vec4d coeff = get_bezier_coeff(t);
    Vec4d d_coeff = get_bezier_d_coeff(t);
    Vec4d dd_coeff = get_bezier_dd_coeff(t);
    const MatxXd& p1 = trajectory.keyframes[idx].pos;
    const MatxXd& p2 = trajectory.keyframes[idx + 1].pos;
    const MatxXd& v1 = trajectory.keyframes[idx].vel;
    const MatxXd& v2 = trajectory.keyframes[idx + 1].vel;
    double T = trajectory.T_between[idx];
    double dtau = info.weight * T;
    MatxXd v_p =
        coeff(0) * p1 + coeff(1) * p2 + T * coeff(2) * v1 + T * coeff(3) * v2;
    MatxXd x = lbs_model.lbs_weights * v_p;
    pd_solver.localStep(x, data.elements);
    MatxXd JP = pd_solver.J * pd_solver.P;
    rhs += dtau * JacobianFres[s].transpose() * JP;
  }
  rhs = rhs - sum_rhs_hessian * arg_selection.C;
}

void BezierLBS2D::global_step() {
  Eigen::MatrixXd new_arg = solver.solve(rhs);

  MatxXd new_full_arg = arg_selection.C + arg_selection.S * new_arg;
  for (int i = 0; i < n_keyframes; i++) {
    trajectory.keyframes[i].pos =
        new_full_arg.block(i * n_controls * 2, 0, n_controls, 2);
    trajectory.keyframes[i].vel =
        new_full_arg.block(i * n_controls * 2 + n_controls, 0, n_controls, 2);
  }
}

}  // namespace aphys
