#include "simTF/optimize/newton.h"

#include <ArmorerPhys/glmath.h>
#include <iostream>

#include "simTF/spline.h"
#include "simTF/optimize/sample.h"
#include "simTF/optimize/argPV.h"

namespace aphys {

template <int dim>
void compute_force_residual(Vecxd& f, const Vecxd& material_jacobian,
                            const MatxXd& v_acce, const Vecxd& vert_mass,
                            const double currT) {
  int n_verts = vert_mass.size();
  assert(f.size() == n_verts * dim);
  for (int i = 0; i < n_verts; i++) {
    Vecxd force = material_jacobian.segment<dim>(i * dim);
    Vecxd residual =
        vert_mass(i) * v_acce.row(i).transpose() / currT / currT + force;
    f.segment<dim>(i * dim) = residual;
  }
}

template void compute_force_residual<2>(Vecxd&, const Vecxd&, const MatxXd&,
                                        const Vecxd&, const double);
template void compute_force_residual<3>(Vecxd&, const Vecxd&, const MatxXd&,
                                        const Vecxd&, const double);

//! remove v_p, v_acce from the function signature
template <int dim>
double compute_Wsp(const SampleBatch& batch,            //
                   const Vecxd& vert_mass,              //
                   const SplineTrajectory& trajectory,  //
                   const Energy_Jacobian_Func& energy_jacobian_func) {
  double Wsp = 0.0;
  int n_verts = trajectory.n_verts;
  MatxXd v_p(n_verts, dim);
  MatxXd v_acce(n_verts, dim);
  Vecxd f(n_verts * dim);
  Vecxd energy_jacobian(n_verts * dim);
  for (int s = 0; s < batch.n_samples; s++) {
    const SampleInfo& info = batch.samples[s];
    double currT = trajectory.T_between[info.keyframe_idx];
    sample_dynamic_status(info, trajectory, v_p, v_acce);
    energy_jacobian_func(v_p, energy_jacobian);

    double W = 0.0;
    compute_force_residual<dim>(f, energy_jacobian, v_acce, vert_mass, currT);
    W += 0.5 * f.transpose() * f;
    Wsp += W * info.weight * trajectory.T_between[info.keyframe_idx];
  }
  return Wsp;
}

template double compute_Wsp<2>(const SampleBatch&, const Vecxd&,
                               const SplineTrajectory&,
                               const Energy_Jacobian_Func&);
template double compute_Wsp<3>(const SampleBatch&, const Vecxd&,
                               const SplineTrajectory&,
                               const Energy_Jacobian_Func&);

template <int dim>
void precompute_hessian_batch(const SampleBatch& batch, const ArgIdxPV& argPV,
                              const std::vector<double> T_between,
                              const Vecxd& vert_mass, const SparseMatd& L_ext,
                              std::vector<SparseMatd>& hessians,
                              SparseMatd& sum_hessian) {
  int n_verts = vert_mass.size();
  int n_args = argPV.n_argP + argPV.n_argV;
  sum_hessian.resize(n_args * dim, n_args * dim);
  sum_hessian.setZero();

  for (const SampleInfo& info : batch.samples) {
    SparseMatd B(n_verts * dim, n_args * dim);
    B.setZero();
    SparseMatd MBdd(n_verts * dim, n_args * dim);
    MBdd.setZero();

    int idx = info.keyframe_idx;
    double T = T_between[idx];
    double t = info.t;

    for (int i = 0; i < n_verts; i++) {
      int p1_idx = argPV.keyframe_idx[idx].P_idx_in_arg(i);
      int p2_idx = argPV.keyframe_idx[idx + 1].P_idx_in_arg(i);
      int v1_idx = argPV.keyframe_idx[idx].V_idx_in_arg(i);
      int v2_idx = argPV.keyframe_idx[idx + 1].V_idx_in_arg(i);

      double mass_tilde = vert_mass(i) / T / T;

      Vec4d coeff = get_bezier_coeff(t);
      Vec4d dd_coeff = get_bezier_dd_coeff(t);

      Eigen::VectorXd I = Eigen::VectorXd::Ones(dim);

      if (p1_idx != -1) {
        int arg_idx = p1_idx;
        set_sparse_block_from_diagnol(B, coeff(0) * I, i * dim, arg_idx * dim);
        set_sparse_block_from_diagnol(MBdd, mass_tilde * dd_coeff(0) * I,
                                      i * dim, arg_idx * dim);
      }
      if (p2_idx != -1) {
        int arg_idx = p2_idx;
        set_sparse_block_from_diagnol(B, coeff(1) * I, i * dim, arg_idx * dim);
        set_sparse_block_from_diagnol(MBdd, mass_tilde * dd_coeff(1) * I,
                                      i * dim, arg_idx * dim);
      }
      if (v1_idx != -1) {
        int arg_idx = v1_idx;
        set_sparse_block_from_diagnol(B, T * coeff(2) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(MBdd, mass_tilde * T * dd_coeff(2) * I,
                                      i * dim, arg_idx * dim);
      }
      if (v2_idx != -1) {
        int arg_idx = v2_idx;
        set_sparse_block_from_diagnol(B, T * coeff(3) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(MBdd, mass_tilde * T * dd_coeff(3) * I,
                                      i * dim, arg_idx * dim);
      }
    }

    double dtau = info.weight * T_between[info.keyframe_idx];
    SparseMatd sample_J_arg = L_ext * B;
    sample_J_arg = MBdd + sample_J_arg;
    hessians.push_back(sample_J_arg.transpose());
    sum_hessian += dtau * hessians.back() * sample_J_arg;
  }
  sum_hessian.makeCompressed();
}

template void precompute_hessian_batch<2>(const SampleBatch&, const ArgIdxPV&,
                                          const std::vector<double>,
                                          const Vecxd&, const SparseMatd&,
                                          std::vector<SparseMatd>&,
                                          SparseMatd&);
template void precompute_hessian_batch<3>(const SampleBatch&, const ArgIdxPV&,
                                          const std::vector<double>,
                                          const Vecxd&, const SparseMatd&,
                                          std::vector<SparseMatd>&,
                                          SparseMatd&);

template <int dim>
void precompute_eulerized_hessian_batch(const SampleBatch& batch,
                                        const ArgIdxPV& argPV,
                                        const std::vector<double> T_between,
                                        const Vecxd& vert_mass,
                                        const SparseMatd& L_ext,
                                        std::vector<SparseMatd>& hessians,
                                        std::vector<SparseMatd>& hessians_euler,
                                        SparseMatd& sum_hessian, double h) {
  int n_verts = vert_mass.size();
  int n_args = argPV.n_argP + argPV.n_argV;
  sum_hessian.resize(n_args * dim, n_args * dim);
  hessians.clear();

  DiagMatxXd M(n_verts * dim);
  set_diag_matrix(vert_mass, M, dim);

  for (const SampleInfo& info : batch.samples) {
    SparseMatd B(n_verts * dim, n_args * dim);
    SparseMatd MBdd(n_verts * dim, n_args * dim);

    int idx = info.keyframe_idx;
    double T = T_between[idx];
    double t = info.t;

    for (int i = 0; i < n_verts; i++) {
      int p1_idx = argPV.keyframe_idx[idx].P_idx_in_arg(i);
      int p2_idx = argPV.keyframe_idx[idx + 1].P_idx_in_arg(i);
      int v1_idx = argPV.keyframe_idx[idx].V_idx_in_arg(i);
      int v2_idx = argPV.keyframe_idx[idx + 1].V_idx_in_arg(i);

      double mass_tilde = vert_mass(i) / T / T;

      Vec4d coeff = get_bezier_coeff(t);
      Vec4d dd_coeff = get_bezier_dd_coeff(t);

      Eigen::VectorXd I = Eigen::VectorXd::Ones(dim);

      if (p1_idx != -1) {
        int arg_idx = p1_idx;
        set_sparse_block_from_diagnol(B, coeff(0) * I, i * dim, arg_idx * dim);
        set_sparse_block_from_diagnol(MBdd, mass_tilde * dd_coeff(0) * I,
                                      i * dim, arg_idx * dim);
      }
      if (p2_idx != -1) {
        int arg_idx = p2_idx;
        set_sparse_block_from_diagnol(B, coeff(1) * I, i * dim, arg_idx * dim);
        set_sparse_block_from_diagnol(MBdd, mass_tilde * dd_coeff(1) * I,
                                      i * dim, arg_idx * dim);
      }
      if (v1_idx != -1) {
        int arg_idx = v1_idx;
        set_sparse_block_from_diagnol(B, T * coeff(2) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(MBdd, mass_tilde * T * dd_coeff(2) * I,
                                      i * dim, arg_idx * dim);
      }
      if (v2_idx != -1) {
        int arg_idx = v2_idx;
        set_sparse_block_from_diagnol(B, T * coeff(3) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(MBdd, mass_tilde * T * dd_coeff(3) * I,
                                      i * dim, arg_idx * dim);
      }
    }
    double dtau = info.weight * T_between[info.keyframe_idx];
    SparseMatd sample_J_arg = L_ext * B;
    sample_J_arg = MBdd + sample_J_arg;
    hessians.push_back(sample_J_arg.transpose());

    hessians_euler.push_back(1.0 / h / h * B.transpose() * M);

    sum_hessian += dtau * hessians.back() * sample_J_arg;
    sum_hessian += dtau * hessians_euler.back() * B;
  }
  sum_hessian.makeCompressed();
}

template void precompute_eulerized_hessian_batch<3>(
    const SampleBatch& batch, const ArgIdxPV& argPV,
    const std::vector<double> T_between, const Vecxd& vert_mass,
    const SparseMatd& L_ext, std::vector<SparseMatd>& hessians,
    std::vector<SparseMatd>& hessians_euler, SparseMatd& sum_hessian, double h);

template void precompute_eulerized_hessian_batch<2>(
    const SampleBatch& batch, const ArgIdxPV& argPV,
    const std::vector<double> T_between, const Vecxd& vert_mass,
    const SparseMatd& L_ext, std::vector<SparseMatd>& hessians,
    std::vector<SparseMatd>& hessians_euler, SparseMatd& sum_hessian, double h);

template <int dim>
double summarize_derivative_PV_with_pre(
    const SampleBatch& batch,                         //
    const ArgIdxPV& argPV,                            //
    const SplineTrajectory& trajectory,               //
    const Vecxd& vert_mass,                           //
    Vecxd& sum_J_arg,                                 //
    std::vector<SparseMatd>& hessians,                //
    const Energy_Jacobian_Func& energy_jacobian_func  //
) {
  const int n_samples = batch.n_samples;
  const int n_verts = trajectory.n_verts;
  const int n_args = argPV.n_argP + argPV.n_argV;
  sum_J_arg.setZero();
  double Wsp = 0.0;

  MatxXd v_p(n_verts, dim);
  MatxXd v_acce(n_verts, dim);
  Vecxd f(n_verts * dim);
  Vecxd energy_jacobian(n_verts * dim);

  int idx = 0;
  for (const SampleInfo& info : batch.samples) {
    double currT = trajectory.T_between[info.keyframe_idx];
    sample_dynamic_status(info, trajectory, v_p, v_acce);
    energy_jacobian_func(v_p, energy_jacobian);
    compute_force_residual<dim>(f, energy_jacobian, v_acce, vert_mass, currT);
    double dtau = info.weight * trajectory.T_between[info.keyframe_idx];
    Wsp += dtau * 0.5 * f.dot(f);
    sum_J_arg += dtau * hessians[idx] * f;
    idx++;
  }
  return Wsp;
}

template double summarize_derivative_PV_with_pre<2>(
    const SampleBatch&, const ArgIdxPV&, const SplineTrajectory&, const Vecxd&,
    Vecxd&, std::vector<SparseMatd>&, const Energy_Jacobian_Func&);
template double summarize_derivative_PV_with_pre<3>(
    const SampleBatch&, const ArgIdxPV&, const SplineTrajectory&, const Vecxd&,
    Vecxd&, std::vector<SparseMatd>&, const Energy_Jacobian_Func&);

template <int dim>
double summarize_eularized_jacobian_PV_with_pre(
    const SampleBatch& batch,                          //
    const ArgIdxPV& argPV,                             //
    const SplineTrajectory& trajectory,                //
    const Vecxd& vert_mass,                            //
    Vecxd& sum_J_arg,                                  //
    std::vector<SparseMatd>& hessians,                 //
    std::vector<SparseMatd>& hessians_euler,           //
    const Energy_Jacobian_Func& energy_jacobian_func,  //
    const double h,                                    //
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

  for (const SampleInfo& info : batch.samples) {
    double currT = trajectory.T_between[info.keyframe_idx];
    sample_dynamic_status(info, trajectory, v_p, v_acce);
    sample_dynamic_status(info, pred_trajectory, pred_v, pred_acce);
    energy_jacobian_func(v_p, energy_jacobian);
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

template double summarize_eularized_jacobian_PV_with_pre<2>(
    const SampleBatch&, const ArgIdxPV&, const SplineTrajectory&, const Vecxd&,
    Vecxd&, std::vector<SparseMatd>&, std::vector<SparseMatd>&,
    const Energy_Jacobian_Func&, const double, const SplineTrajectory&);
template double summarize_eularized_jacobian_PV_with_pre<3>(
    const SampleBatch&, const ArgIdxPV&, const SplineTrajectory&, const Vecxd&,
    Vecxd&, std::vector<SparseMatd>&, std::vector<SparseMatd>&,
    const Energy_Jacobian_Func&, const double, const SplineTrajectory&);

template <int dim>
void apply_arg_in_solver(const ArgIdxPV& argPV,               //
                         const SplineTrajectory& trajectory,  //
                         const Vecxd& curr_arg, SplineTrajectory& traj_new) {
  for (int k = 0; k < trajectory.n_keyframes; k++) {
    const KeyframeNode& keyframe = trajectory.keyframes[k];
    KeyframeNode& new_keyframe = traj_new.keyframes[k];
    const KeyframeIdxPV& idx = argPV.keyframe_idx[k];
    for (int i = 0; i < trajectory.n_verts; i++) {
      if (idx.P_idx_in_arg(i) != -1)
        new_keyframe.pos.row(i) =
            curr_arg.segment<dim>(idx.P_idx_in_arg(i) * dim).transpose();
      if (idx.V_idx_in_arg(i) != -1)
        new_keyframe.vel.row(i) =
            curr_arg.segment<dim>(idx.V_idx_in_arg(i) * dim).transpose();
    }
  }
}

template void apply_arg_in_solver<2>(const ArgIdxPV&, const SplineTrajectory&,
                                     const Vecxd&, SplineTrajectory&);
template void apply_arg_in_solver<3>(const ArgIdxPV&, const SplineTrajectory&,
                                     const Vecxd&, SplineTrajectory&);

void apply_arg_in_solver(const ArgIdxPV& argPV,               //
                         const SplineTrajectory& trajectory,  //
                         const MatxXd& curr_arg, SplineTrajectory& traj_new) {
  for (int k = 0; k < trajectory.n_keyframes; k++) {
    const KeyframeNode& keyframe = trajectory.keyframes[k];
    KeyframeNode& new_keyframe = traj_new.keyframes[k];
    const KeyframeIdxPV& idx = argPV.keyframe_idx[k];
    for (int i = 0; i < trajectory.n_verts; i++) {
      if (idx.P_idx_in_arg(i) != -1)
        new_keyframe.pos.row(i) = curr_arg.row(idx.P_idx_in_arg(i)).transpose();
      if (idx.V_idx_in_arg(i) != -1)
        new_keyframe.vel.row(i) = curr_arg.row(idx.V_idx_in_arg(i)).transpose();
    }
  }
}

template <int dim>
void summarize_derivative_T(const SampleBatch& batch,            //
                            const SplineTrajectory& trajectory,  //
                            const Vecxd& vert_mass,              //
                            const Energy_Jacobian_Func& energy_jacobian_func,
                            const SparseMatd& L_ext, Vecxd& sum_J_T,
                            Vecxd& sum_H_T) {
  const int n_verts = trajectory.n_verts;
  const int n_intervals = trajectory.n_keyframes - 1;
  sum_J_T.resize(n_intervals);
  sum_J_T.setZero();
  sum_H_T.resize(n_intervals);
  sum_H_T.setZero();

  MatxXd v_p(n_verts, dim);
  MatxXd v_acce(n_verts, dim);
  MatxXd Bvq(n_verts, dim);
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
    Vec4d dd_coeff = get_bezier_dd_coeff(t);
    const MatxXd& p1 = trajectory.keyframes[sdx].pos;
    const MatxXd& p2 = trajectory.keyframes[sdx + 1].pos;
    const MatxXd& v1 = trajectory.keyframes[sdx].vel;
    const MatxXd& v2 = trajectory.keyframes[sdx + 1].vel;
    double w = info.weight;

    v_p = coeff(0) * p1 + coeff(1) * p2 + T * coeff(2) * v1 + T * coeff(3) * v2;
    v_acce = dd_coeff(0) * p1 + dd_coeff(1) * p2 + T * dd_coeff(2) * v1 +
             T * dd_coeff(3) * v2;
    Bvq = coeff(2) * v1 + coeff(3) * v2;
    Bvddq = dd_coeff(2) * v1 + dd_coeff(3) * v2;

    Vecxd v_p_vec = aphys::Vecxd::Map(v_p.data(), v_p.size());
    Vecxd v_acce_vec = aphys::Vecxd::Map(v_acce.data(), v_acce.size());
    Vecxd Bvq_vec = aphys::Vecxd::Map(Bvq.data(), Bvq.size());
    Vecxd Bvddq_vec = aphys::Vecxd::Map(Bvddq.data(), Bvddq.size());

    compute_force_residual<dim>(f, energy_jacobian, v_acce, vert_mass, T);

    Vecxd dFdT = -2.0 * M * v_acce_vec / T / T / T + M * Bvddq_vec / T / T -
                 L_ext * Bvq_vec;
    Vecxd d2FdT2 =
        6.0 * M * v_acce_vec / T / T / T / T - 4.0 * M * Bvddq_vec / T / T / T;

    double dWdT = 0.5 * w * f.dot(f) + w * T * dFdT.dot(f);
    double d2WdT2 = 2.0 * w * dFdT.dot(f) + w * w * T * d2FdT2.dot(f) +
                    w * T * dFdT.dot(dFdT);

    sum_J_T(sdx) += dWdT;
    sum_H_T(sdx) += d2WdT2;
  }
}

template void summarize_derivative_T<2>(const SampleBatch&,
                                        const SplineTrajectory&, const Vecxd&,
                                        const Energy_Jacobian_Func&,
                                        const SparseMatd&, Vecxd&, Vecxd&);
template void summarize_derivative_T<3>(const SampleBatch&,
                                        const SplineTrajectory&, const Vecxd&,
                                        const Energy_Jacobian_Func&,
                                        const SparseMatd&, Vecxd&, Vecxd&);

template <int dim>
void T_line_search(const SampleBatch& batch,      //
                   SplineTrajectory& trajectory,  //
                   const Vecxd& gradient_T_direc, const Vecxd& search_T_direc,
                   const Vecxd& vert_mass,
                   const Energy_Jacobian_Func& energy_jacobian_func) {
  double alpha = 1.0;
  double beta = 0.5;
  double gamma = 0.03;
  double ddv = -gradient_T_direc.dot(search_T_direc);

  int n_invervals = trajectory.n_keyframes - 1;
  std::vector<double> origin_T_between(n_invervals);
  for (int i = 0; i < n_invervals; i++) {
    origin_T_between[i] = trajectory.T_between[i];
  }

  auto energy_func = [&](const SplineTrajectory& traj) -> double {
    return compute_Wsp<dim>(batch, vert_mass, traj, energy_jacobian_func);
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
    if (iter > 2) {
      std::cout << "Line search failed" << std::endl;
      apply_args(Vecxd::Zero(n_invervals));
      break;
    }
  }

  std::cout << "Line search step size: " << alpha << std::endl;
}

template void T_line_search<2>(const SampleBatch&, SplineTrajectory&,
                               const Vecxd&, const Vecxd&, const Vecxd&,
                               const Energy_Jacobian_Func&);
template void T_line_search<3>(const SampleBatch&, SplineTrajectory&,
                               const Vecxd&, const Vecxd&, const Vecxd&,
                               const Energy_Jacobian_Func&);

}  // namespace aphys
