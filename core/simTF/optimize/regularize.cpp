#include "simTF/optimize/regularize.h"
#include "simTF/optimize/sample.h"
#include "simTF/optimize/newton.h"

#include <iostream>
#include <ArmorerPhys/glmath.h>

namespace aphys {

template <int dim>
Regularizer<dim>::Regularizer(int n_verts) {
  int n_args = n_verts * 4;
  Av.resize(0, 0);
  Av.resize(n_verts, n_args);
  Bx.resize(0, 0);
  Bx.resize(n_verts, n_args);
  Bv.resize(0, 0);
  Bv.resize(n_verts, n_args);
  Cv.resize(0, 0);
  Cv.resize(n_verts, n_args);
  Dx.resize(0, 0);
  Dx.resize(n_verts, n_args);

  auto x1 = [&](int i) -> int { return i; };
  auto v1 = [&](int i) -> int { return i + n_verts; };
  auto x2 = [&](int i) -> int { return i + n_verts * 2; };
  auto v2 = [&](int i) -> int { return i + n_verts * 3; };

  for (int i = 0; i < n_verts; i++) {
    Av.insert(i, v1(i)) = 1.0;

    Bx.insert(i, x1(i)) = -1.0;
    Bx.insert(i, x2(i)) = 1.0;
    Bv.insert(i, v1(i)) = -1.0;
    Bv.insert(i, v2(i)) = -1.0;

    Cv.insert(i, v2(i)) = 1.0;

    Dx.insert(i, x1(i)) = -1.0;
    Dx.insert(i, x2(i)) = 1.0;
  }

  A = Av.transpose() * Av;
  C = Cv.transpose() * Cv;
  D = -Dx.transpose() * Dx / 3.0;
}

template <int dim>
void Regularizer<dim>::compute_regularize_gradient(
    const SplineTrajectory& trajectory,  //
    SparseMatd& sum_HR) {
  const int n_verts = trajectory.n_verts;
  const int n_keyframes = trajectory.n_keyframes;
  const int n_intervals = n_keyframes - 1;
  const int n_args = 2 * n_keyframes * n_verts;
  sum_HR.resize(n_args, n_args);

  for (int k = 0; k < n_intervals; k++) {
    const double T = trajectory.T_between[k];

    SparseMatd B = Bx + T * Bv;
    B = B.transpose() * B;

    SparseMatd U = A * T * T + B + C * T * T + D;

    add_sparse_block(sum_HR, U, k * 2 * n_verts, k * 2 * n_verts);
  }
}

template <int dim>
void Regularizer<dim>::add_PD_gradient(const SplineTrajectory& trajectory,  //
                                       SparseMatd& L,                       //
                                       SparseMatd& sum_HR) {
  const int n_verts = trajectory.n_verts;
  const int n_keyframes = trajectory.n_keyframes;

  for (int k = 0; k < n_keyframes; k++) {
    add_sparse_block(sum_HR, L, k * 2 * n_verts, k * 2 * n_verts);
  }
}

template <int dim>
void Regularizer<dim>::regularize(const SplineTrajectory& trajectory,  //
                                  SparseMatd& sum_HR, MatxXd& rhs,
                                  const ArgIdxPV& arg_pv, MatxXd& new_arg) {
  const int n_verts = trajectory.n_verts;
  const int n_keyframes = trajectory.n_keyframes;
  const int n_args = arg_pv.n_argP + arg_pv.n_argV;

  SparseMatd S(n_keyframes * n_verts * 2, n_args);
  MatxXd C(n_keyframes * n_verts * 2, dim);
  C.setZero();

  for (int k = 0; k < n_keyframes; k++) {
    for (int i = 0; i < n_verts; i++) {
      int idx = arg_pv.keyframe_idx[k].P_idx_in_arg(i);
      if (!trajectory.keyframes[k].p_fix(i)) {
        S.insert(k * n_verts * 2 + i, idx) = 1.0;
      } else {
        C.row(k * n_verts * 2 + i) = trajectory.keyframes[k].pos.row(i);
      }
      idx = arg_pv.keyframe_idx[k].V_idx_in_arg(i);
      if (!trajectory.keyframes[k].v_fix(i)) {
        S.insert(k * n_verts * 2 + i + n_verts, idx) = 1.0;
      } else {
        C.row(k * n_verts * 2 + i + n_verts) =
            trajectory.keyframes[k].vel.row(i);
      }
    }
  }

  // x = S * arg + C
  SparseMatd lhs = S.transpose() * sum_HR * S;
  MatxXd rhs_new = S.transpose() * (rhs - sum_HR * C);

  solver.analyzePattern(lhs);
  solver.factorize(lhs);
  new_arg = solver.solve(rhs_new);
}

template <int dim>
void Regularizer<dim>::recompute_reference_deformationP(
    const SampleBatch& batch, const SplineTrajectory& trajectory,
    ProjectiveDynamicsSolver<dim>& pd_solver, const EleMat& elements,
    std::vector<MatxXd>& newP) {
  newP.clear();
  int n_verts = trajectory.n_verts;
  MatxXd v_p(n_verts, dim);
  MatxXd v_acce(n_verts, dim);
  for (int s = 0; s < batch.n_samples; s++) {
    const SampleInfo& info = batch.samples[s];
    double currT = trajectory.T_between[info.keyframe_idx];
    sample_dynamic_status(info, trajectory, v_p, v_acce);
    MatxXd new_P;
    pd_solver.recompute_reference(v_p, elements, new_P);
    newP.push_back(new_P);
  }
}

template struct Regularizer<2>;
template struct Regularizer<3>;

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
    const SplineTrajectory& pred_trajectory) {
  const int n_samples = batch.n_samples;
  const int n_verts = trajectory.n_verts;
  const int n_args = argPV.n_argP + argPV.n_argV;
  sum_J_arg.setZero();
  double Wsp = 0.0;
  int idx = 0;

  MatxXd v_p(n_verts, dim);
  MatxXd pred_v(n_verts, dim);
  MatxXd v_acce(n_verts, dim);
  MatxXd pred_acce(n_verts, dim);
  Vecxd f(n_verts * dim);
  Vecxd energy_jacobian(n_verts * dim);

  for (int s = 0; s < n_samples; s++) {
    const SampleInfo& info = batch.samples[s];
    double currT = trajectory.T_between[info.keyframe_idx];
    sample_dynamic_status(info, trajectory, v_p, v_acce);
    sample_dynamic_status(info, pred_trajectory, pred_v, pred_acce);
    energy_jacobian_funcs[s](v_p, energy_jacobian);
    compute_force_residual<dim>(f, energy_jacobian, v_acce, vert_mass, currT);

    double dtau = info.weight * trajectory.T_between[info.keyframe_idx];
    Wsp += dtau * 0.5 * f.dot(f);
    v_p = v_p - pred_v;
    Vecxd dx_vec = aphys::Vecxd::Map(v_p.data(), v_p.size());
    sum_J_arg += dtau * hessians[idx] * f + dtau * hessians_euler[idx] * dx_vec;
    idx++;
  }
  return Wsp;
}

template double summarize_regularized_eularized_jacobian_PV_with_pre<2>(
    const SampleBatch&, const ArgIdxPV&, const SplineTrajectory&, const Vecxd&,
    Vecxd&, std::vector<SparseMatd>&, std::vector<SparseMatd>&,
    const std::vector<Energy_Jacobian_Func>&, const double,
    const SplineTrajectory&);
template double summarize_regularized_eularized_jacobian_PV_with_pre<3>(
    const SampleBatch&, const ArgIdxPV&, const SplineTrajectory&, const Vecxd&,
    Vecxd&, std::vector<SparseMatd>&, std::vector<SparseMatd>&,
    const std::vector<Energy_Jacobian_Func>&, const double,
    const SplineTrajectory&);

}  // namespace aphys
