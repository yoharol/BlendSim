#include "simTF/optimize/damping_version.h"

#include <ArmorerPhys/glmath.h>
#include <iostream>

#include "simTF/spline.h"
#include "simTF/optimize/sample.h"
#include "simTF/optimize/argPV.h"

namespace aphys {

void sample_damped_dynamic_status(  //
    const SampleInfo& sample_info,
    const SplineTrajectory& trajectory,         //
    MatxXd& v_p, MatxXd& v_vel, MatxXd& v_acce  //
) {
  const int idx = sample_info.keyframe_idx;
  const double t = sample_info.t;
  Vec4d coeff = get_bezier_coeff(t);
  Vec4d d_coeff = get_bezier_d_coeff(t);
  Vec4d dd_coeff = get_bezier_dd_coeff(t);
  const MatxXd& p1 = trajectory.keyframes[idx].pos;
  const MatxXd& p2 = trajectory.keyframes[idx + 1].pos;
  const MatxXd& v1 = trajectory.keyframes[idx].vel;
  const MatxXd& v2 = trajectory.keyframes[idx + 1].vel;
  const double T = trajectory.T_between[idx];

  v_p = coeff(0) * p1 + coeff(1) * p2 + T * coeff(2) * v1 + T * coeff(3) * v2;
  v_vel = d_coeff(0) * p1 + d_coeff(1) * p2 + T * d_coeff(2) * v1 +
          T * d_coeff(3) * v2;
  v_acce = dd_coeff(0) * p1 + dd_coeff(1) * p2 + T * dd_coeff(2) * v1 +
           T * dd_coeff(3) * v2;
}

template <int dim>
void compute_damped_force_residual(Vecxd& f, const Vecxd& material_jacobian,
                                   const MatxXd& v_vel, const MatxXd& v_acce,
                                   const Vecxd& vert_mass, const double currT,
                                   const double damping_alpha) {
  int n_verts = vert_mass.size();
  assert(f.size() == n_verts * dim);
  for (int i = 0; i < n_verts; i++) {
    Vecxd force = material_jacobian.segment<dim>(i * dim);
    Vecxd residual =
        vert_mass(i) * v_acce.row(i).transpose() / currT / currT +
        damping_alpha * vert_mass(i) * v_vel.row(i).transpose() / currT + force;
    f.segment<dim>(i * dim) = residual;
  }
}

template void compute_damped_force_residual<2>(Vecxd&, const Vecxd&,
                                               const MatxXd&, const MatxXd&,
                                               const Vecxd&, const double,
                                               const double);
template void compute_damped_force_residual<3>(Vecxd&, const Vecxd&,
                                               const MatxXd&, const MatxXd&,
                                               const Vecxd&, const double,
                                               const double);

template <int dim>
double compute_damped_Wsp(const SampleBatch& batch,            //
                          const Vecxd& vert_mass,              //
                          const SplineTrajectory& trajectory,  //
                          const Energy_Jacobian_Func& energy_jacobian_func,
                          const double damping_alpha) {
  double Wsp = 0.0;
  int n_verts = trajectory.n_verts;
  MatxXd v_p(n_verts, dim);
  MatxXd v_vel(n_verts, dim);
  MatxXd v_acce(n_verts, dim);
  Vecxd f(n_verts * dim);
  Vecxd energy_jacobian(n_verts * dim);
  for (int s = 0; s < batch.n_samples; s++) {
    const SampleInfo& info = batch.samples[s];
    double currT = trajectory.T_between[info.keyframe_idx];
    sample_damped_dynamic_status(info, trajectory, v_p, v_vel, v_acce);
    energy_jacobian_func(v_p, energy_jacobian);

    double W = 0.0;
    compute_damped_force_residual<dim>(f, energy_jacobian, v_vel, v_acce,
                                       vert_mass, currT, damping_alpha);
    W += 0.5 * f.squaredNorm();
    Wsp += W * info.weight * trajectory.T_between[info.keyframe_idx];
  }
  return Wsp;
}

template double compute_damped_Wsp<2>(const SampleBatch&, const Vecxd&,
                                      const SplineTrajectory&,
                                      const Energy_Jacobian_Func&,
                                      const double);
template double compute_damped_Wsp<3>(const SampleBatch&, const Vecxd&,
                                      const SplineTrajectory&,
                                      const Energy_Jacobian_Func&,
                                      const double);

template <int dim>
void precompute_damped_eulerized_hessian_batch(
    const SampleBatch& batch, const ArgIdxPV& argPV,
    const std::vector<double> T_between, const Vecxd& vert_mass,
    const SparseMatd& L_ext, const double damping_alpha,
    std::vector<SparseMatd>& hessians, std::vector<SparseMatd>& hessians_euler,
    SparseMatd& sum_hessian, double h) {
  int n_verts = vert_mass.size();
  int n_args = argPV.n_argP + argPV.n_argV;
  sum_hessian.resize(n_args * dim, n_args * dim);
  sum_hessian.setZero();
  hessians.clear();
  hessians_euler.clear();

  DiagMatxXd M(n_verts * dim);
  set_diag_matrix(vert_mass, M, dim);

  for (const SampleInfo& info : batch.samples) {
    SparseMatd B(n_verts * dim, n_args * dim);
    SparseMatd MBdd(n_verts * dim, n_args * dim);
    SparseMatd aMBd(n_verts * dim, n_args * dim);

    int idx = info.keyframe_idx;
    double T = T_between[idx];
    double t = info.t;

    for (int i = 0; i < n_verts; i++) {
      int p1_idx = argPV.keyframe_idx[idx].P_idx_in_arg(i);
      int p2_idx = argPV.keyframe_idx[idx + 1].P_idx_in_arg(i);
      int v1_idx = argPV.keyframe_idx[idx].V_idx_in_arg(i);
      int v2_idx = argPV.keyframe_idx[idx + 1].V_idx_in_arg(i);

      double mass_t = damping_alpha * vert_mass(i) / T;
      double mass_tilde = vert_mass(i) / T / T;

      Vec4d coeff = get_bezier_coeff(t);
      Vec4d d_coeff = get_bezier_d_coeff(t);
      Vec4d dd_coeff = get_bezier_dd_coeff(t);

      Eigen::VectorXd I = Eigen::VectorXd::Ones(dim);

      if (p1_idx != -1) {
        int arg_idx = p1_idx;
        set_sparse_block_from_diagnol(B, coeff(0) * I, i * dim, arg_idx * dim);
        set_sparse_block_from_diagnol(aMBd, mass_t * d_coeff(0) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(MBdd, mass_tilde * dd_coeff(0) * I,
                                      i * dim, arg_idx * dim);
      }
      if (p2_idx != -1) {
        int arg_idx = p2_idx;
        set_sparse_block_from_diagnol(B, coeff(1) * I, i * dim, arg_idx * dim);
        set_sparse_block_from_diagnol(aMBd, mass_t * d_coeff(1) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(MBdd, mass_tilde * dd_coeff(1) * I,
                                      i * dim, arg_idx * dim);
      }
      if (v1_idx != -1) {
        int arg_idx = v1_idx;
        set_sparse_block_from_diagnol(B, T * coeff(2) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(aMBd, mass_t * T * d_coeff(2) * I,
                                      i * dim, arg_idx * dim);
        set_sparse_block_from_diagnol(MBdd, mass_tilde * T * dd_coeff(2) * I,
                                      i * dim, arg_idx * dim);
      }
      if (v2_idx != -1) {
        int arg_idx = v2_idx;
        set_sparse_block_from_diagnol(B, T * coeff(3) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(aMBd, mass_t * T * d_coeff(3) * I,
                                      i * dim, arg_idx * dim);
        set_sparse_block_from_diagnol(MBdd, mass_tilde * T * dd_coeff(3) * I,
                                      i * dim, arg_idx * dim);
      }
    }
    double dtau = info.weight * T_between[info.keyframe_idx];
    SparseMatd sample_J_arg = L_ext * B;
    sample_J_arg = MBdd + aMBd + sample_J_arg;
    hessians.push_back(sample_J_arg.transpose());

    hessians_euler.push_back(1.0 / h / h * B.transpose() * M);

    sum_hessian += dtau * hessians.back() * sample_J_arg;
    sum_hessian += dtau * hessians_euler.back() * B;
  }
  sum_hessian.makeCompressed();
}

template void precompute_damped_eulerized_hessian_batch<2>(
    const SampleBatch&, const ArgIdxPV&, const std::vector<double>,
    const Vecxd&, const SparseMatd&, const double, std::vector<SparseMatd>&,
    std::vector<SparseMatd>&, SparseMatd&, double);
template void precompute_damped_eulerized_hessian_batch<3>(
    const SampleBatch&, const ArgIdxPV&, const std::vector<double>,
    const Vecxd&, const SparseMatd&, const double, std::vector<SparseMatd>&,
    std::vector<SparseMatd>&, SparseMatd&, double);

template <int dim>
double summarize_damped_eularized_jacobian_PV_with_pre(
    const SampleBatch& batch,                          //
    const ArgIdxPV& argPV,                             //
    const SplineTrajectory& trajectory,                //
    const Vecxd& vert_mass,                            //
    Vecxd& sum_J_arg,                                  //
    std::vector<SparseMatd>& hessians,                 //
    std::vector<SparseMatd>& hessians_euler,           //
    const Energy_Jacobian_Func& energy_jacobian_func,  //
    const double damping_alpha) {
  const int n_samples = batch.n_samples;
  const int n_verts = trajectory.n_verts;
  const int n_args = argPV.n_argP + argPV.n_argV;
  sum_J_arg.setZero();
  double Wsp = 0.0;
  int idx = 0;

  MatxXd v_p(n_verts, dim);
  MatxXd v_vel(n_verts, dim);
  MatxXd v_acce(n_verts, dim);
  Vecxd f(n_verts * dim);
  Vecxd energy_jacobian(n_verts * dim);

  for (const SampleInfo& info : batch.samples) {
    double currT = trajectory.T_between[info.keyframe_idx];
    sample_damped_dynamic_status(info, trajectory, v_p, v_vel, v_acce);
    energy_jacobian_func(v_p, energy_jacobian);
    compute_damped_force_residual<dim>(f, energy_jacobian, v_vel, v_acce,
                                       vert_mass, currT, damping_alpha);

    double dtau = info.weight * trajectory.T_between[info.keyframe_idx];
    Wsp += dtau * 0.5 * f.dot(f);

    //! hessians_euler can be removed later
    v_p.setZero();
    Vecxd dx_vec = aphys::Vecxd::Map(v_p.data(), v_p.size());
    sum_J_arg += dtau * hessians[idx] * f;
    idx++;
  }
  return Wsp;
}

template double summarize_damped_eularized_jacobian_PV_with_pre<2>(
    const SampleBatch&, const ArgIdxPV&, const SplineTrajectory&, const Vecxd&,
    Vecxd&, std::vector<SparseMatd>&, std::vector<SparseMatd>&,
    const Energy_Jacobian_Func&, const double);
template double summarize_damped_eularized_jacobian_PV_with_pre<3>(
    const SampleBatch&, const ArgIdxPV&, const SplineTrajectory&, const Vecxd&,
    Vecxd&, std::vector<SparseMatd>&, std::vector<SparseMatd>&,
    const Energy_Jacobian_Func&, const double);

int sample_damped_tau_on_trajectory(const double tau,              //
                                    SplineTrajectory& trajectory,  //
                                    MatxXd& pos, MatxXd& vel, MatxXd& acce) {
  int idx = 0;
  bool selected = false;
  for (; idx < trajectory.tau_stamp.size(); idx++) {
    if (trajectory.tau_stamp[idx] <= tau &&
        trajectory.tau_stamp[idx + 1] >= tau) {
      selected = true;
      break;
    }
  }
  assert(selected && "[simTF/spline/builder][sample_on_tau]tau out of range");
  const double T = trajectory.T_between[idx];
  double t = (tau - trajectory.tau_stamp[idx]) / T;
  const MatxXd& p1 = trajectory.keyframes[idx].pos;
  const MatxXd& p2 = trajectory.keyframes[idx + 1].pos;
  const MatxXd& v1 = trajectory.keyframes[idx].vel;
  const MatxXd& v2 = trajectory.keyframes[idx + 1].vel;
  Vec4d coeff = get_bezier_coeff(t);
  pos = coeff(0) * p1 + coeff(1) * p2 + T * coeff(2) * v1 + T * coeff(3) * v2;
  Vec4d dd_coeff = get_bezier_dd_coeff(t);
  acce = dd_coeff(0) * p1 + dd_coeff(1) * p2 + T * dd_coeff(2) * v1 +
         T * dd_coeff(3) * v2;
  Vec4d d_coeff = get_bezier_d_coeff(t);
  vel = d_coeff(0) * p1 + d_coeff(1) * p2 + T * d_coeff(2) * v1 +
        T * d_coeff(3) * v2;
  return idx;
}

template <int dim>
void summarize_damped_derivative_T(
    const SampleBatch& batch,            //
    const SplineTrajectory& trajectory,  //
    const Vecxd& vert_mass,              //
    const Energy_Jacobian_Func& energy_jacobian_func, const SparseMatd& L_ext,
    Vecxd& sum_J_T, Vecxd& sum_H_T, const double damping_alpha) {
  const int n_verts = trajectory.n_verts;
  const int n_intervals = trajectory.n_keyframes - 1;
  sum_J_T.resize(n_intervals);
  sum_J_T.setZero();
  sum_H_T.resize(n_intervals);
  sum_H_T.setZero();

  MatxXd v_p(n_verts, dim);
  MatxXd v_vel(n_verts, dim);
  MatxXd v_acce(n_verts, dim);
  MatxXd Bvq(n_verts, dim);
  MatxXd Bvdq(n_verts, dim);
  MatxXd Bvddq(n_verts, dim);
  Vecxd f(n_verts * dim);
  Vecxd energy_jacobian(n_verts * dim);
  DiagMatxXd M(n_verts * dim);
  set_diag_matrix(vert_mass, M, dim);

  for (const SampleInfo& info : batch.samples) {
    int sdx = info.keyframe_idx;
    double T = trajectory.T_between[sdx];

    double t = info.t;
    Vec4d coeff = get_bezier_coeff(t);
    Vec4d d_coeff = get_bezier_d_coeff(t);
    Vec4d dd_coeff = get_bezier_dd_coeff(t);
    const MatxXd& p1 = trajectory.keyframes[sdx].pos;
    const MatxXd& p2 = trajectory.keyframes[sdx + 1].pos;
    const MatxXd& v1 = trajectory.keyframes[sdx].vel;
    const MatxXd& v2 = trajectory.keyframes[sdx + 1].vel;
    double w = info.weight;

    v_p = coeff(0) * p1 + coeff(1) * p2 + T * coeff(2) * v1 + T * coeff(3) * v2;
    v_vel = d_coeff(0) * p1 + d_coeff(1) * p2 + T * d_coeff(2) * v1 +
            T * d_coeff(3) * v2;
    v_acce = dd_coeff(0) * p1 + dd_coeff(1) * p2 + T * dd_coeff(2) * v1 +
             T * dd_coeff(3) * v2;
    Bvq = coeff(2) * v1 + coeff(3) * v2;
    Bvdq = d_coeff(2) * v1 + d_coeff(3) * v2;
    Bvddq = dd_coeff(2) * v1 + dd_coeff(3) * v2;

    Vecxd v_p_vec = aphys::Vecxd::Map(v_p.data(), v_p.size());
    Vecxd v_vel_vec = aphys::Vecxd::Map(v_vel.data(), v_vel.size());
    Vecxd v_acce_vec = aphys::Vecxd::Map(v_acce.data(), v_acce.size());
    Vecxd Bvq_vec = aphys::Vecxd::Map(Bvq.data(), Bvq.size());
    Vecxd Bvdq_vec = aphys::Vecxd::Map(Bvdq.data(), Bvdq.size());
    Vecxd Bvddq_vec = aphys::Vecxd::Map(Bvddq.data(), Bvddq.size());

    energy_jacobian_func(v_p, energy_jacobian);
    compute_damped_force_residual<dim>(f, energy_jacobian, v_vel, v_acce,
                                       vert_mass, T, damping_alpha);

    Vecxd dFdT = -2.0 * M * v_acce_vec / T / T / T + M * Bvddq_vec / T / T -
                 damping_alpha * M * v_vel_vec / T / T +
                 damping_alpha * M * Bvdq_vec / T - L_ext * Bvq_vec;
    Vecxd d2FdT2 = 6.0 * M * v_acce_vec / T / T / T / T -
                   4.0 * M * Bvddq_vec / T / T / T +
                   2.0 * damping_alpha * M * v_vel_vec / T / T -
                   damping_alpha * M * Bvdq_vec / T / T;

    double dWdT = 0.5 * w * f.dot(f) + w * T * dFdT.dot(f);
    double d2WdT2 = 2.0 * w * dFdT.dot(f) + w * w * T * d2FdT2.dot(f) +
                    w * T * dFdT.dot(dFdT);

    sum_J_T(sdx) += dWdT;
    sum_H_T(sdx) += d2WdT2;
  }
}

template void summarize_damped_derivative_T<2>(const SampleBatch&,
                                               const SplineTrajectory&,
                                               const Vecxd&,
                                               const Energy_Jacobian_Func&,
                                               const SparseMatd&, Vecxd&,
                                               Vecxd&, const double);
template void summarize_damped_derivative_T<3>(const SampleBatch&,
                                               const SplineTrajectory&,
                                               const Vecxd&,
                                               const Energy_Jacobian_Func&,
                                               const SparseMatd&, Vecxd&,
                                               Vecxd&, const double);

template <int dim>
void T_damped_line_search(const SampleBatch& batch,      //
                          SplineTrajectory& trajectory,  //
                          const Vecxd& gradient_T_direc,
                          const Vecxd& search_T_direc, const Vecxd& vert_mass,
                          const Energy_Jacobian_Func& energy_jacobian_func,
                          const double damping_alpha) {
  double alpha = 1.0;
  double beta = 0.5;
  double gamma = 0.03;
  double ddv = -gradient_T_direc.dot(search_T_direc);

  int n_invervals = trajectory.n_keyframes - 1;

  for (int i = 0; i < n_invervals; i++) {
    if (search_T_direc(i) < 0)
      alpha = std::min(alpha, -trajectory.T_between[i] / search_T_direc(i));
  }

  std::vector<double> origin_T_between(n_invervals);
  for (int i = 0; i < n_invervals; i++) {
    origin_T_between[i] = trajectory.T_between[i];
  }

  auto energy_func = [&](const SplineTrajectory& traj) -> double {
    double curr_wsp = compute_damped_Wsp<dim>(
        batch, vert_mass, traj, energy_jacobian_func, damping_alpha);
    return curr_wsp;
  };

  double baseline = energy_func(trajectory);

  auto apply_args = [&](const Vecxd& DeltaTime) {
    for (int i = 0; i < n_invervals; i++) {
      trajectory.T_between[i] = origin_T_between[i] + DeltaTime(i);
    }
    update_trajectory_tau(trajectory);
  };

  apply_args(alpha * search_T_direc);
  double first_guess = energy_func(trajectory);

  int iter = 0;

  while (energy_func(trajectory) > baseline + gamma * alpha * ddv) {
    alpha *= beta;
    apply_args(alpha * search_T_direc);
    iter++;
    if (iter > 5) {
      std::cout << "Line search failed" << std::endl;
      apply_args(Vecxd::Zero(n_invervals));
      break;
    }
  }

  std::cout << "Line search step size: " << alpha << std::endl;
}

template void T_damped_line_search<2>(const SampleBatch&, SplineTrajectory&,
                                      const Vecxd&, const Vecxd&, const Vecxd&,
                                      const Energy_Jacobian_Func&,
                                      const double);

template void T_damped_line_search<3>(const SampleBatch&, SplineTrajectory&,
                                      const Vecxd&, const Vecxd&, const Vecxd&,
                                      const Energy_Jacobian_Func&,
                                      const double);

}  // namespace aphys
