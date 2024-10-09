#include "simTF/optimize/subspace.h"

#include <ArmorerPhys/glmath.h>
#include <ArmorerPhys/timer.h>
#include <iostream>

#include "simTF/spline.h"
#include "simTF/optimize/sample.h"
#include "simTF/optimize/argPV.h"
#include "simTF/optimize/damping_version.h"
#include "simTF/PD/pd.h"

namespace aphys {

// MU: M_ext * lbs_ext
double compute_lbs_wsp(const SampleBatch& batch, const int n_verts,
                       const SplineTrajectory& lbs_trajectory,
                       ProjectiveDynamicsSolver<3>& pd_solver,
                       const LBSDataManager& ldm, const double damping_alpha) {
  double Wsp = 0.0;
  int n_controls = lbs_trajectory.n_verts;
  MatxXd c_p(n_controls, 3);
  MatxXd c_v(n_controls, 3);
  MatxXd c_a(n_controls, 3);
  Vecxd material_jacobian(n_verts * 3);

  for (int s = 0; s < batch.n_samples; s++) {
    const SampleInfo& info = batch.samples[s];
    double T = lbs_trajectory.T_between[info.keyframe_idx];
    sample_damped_dynamic_status(info, lbs_trajectory, c_p, c_v, c_a);
    Vecxd c_v_vec = aphys::Vecxd::Map(c_v.data(), c_v.size());
    Vecxd c_a_vec = aphys::Vecxd::Map(c_a.data(), c_a.size());

    MatxXd pos = ldm.U * c_p;
    pd_solver.localStep(pos, ldm.tets);
    MatxXd gradient = ldm.LU * c_p - pd_solver.J * pd_solver.P;
    material_jacobian = Vecxd::Map(gradient.data(), gradient.size());

    double W = 0.0;

    Vecxd f = material_jacobian + ldm.MU_ext * c_a_vec / T / T +
              damping_alpha * ldm.MU_ext * c_v_vec / T;

    Wsp += 0.5 * f.dot(f) * info.weight * T;
    // Wsp += 0.5 * c_v.norm();
  }
  return Wsp;
}

void precompute_lbs_hessian_batch(const SampleBatch& batch,
                                  const ArgIdxPV& argPV,
                                  const std::vector<double> T_between,
                                  const LBSDataManager& ldm,
                                  const double damping_alpha,
                                  std::vector<SparseMatd>& Hessian_P,
                                  SparseMatd& sum_hessian, double h) {
  int dim = 3;
  int n_verts = ldm.n_verts;
  int n_controls = ldm.n_controls;  // control vectors in lbs
  int n_args = argPV.n_argP + argPV.n_argV;
  sum_hessian.resize(0, 0);
  sum_hessian.resize(n_args * dim, n_args * dim);
  Hessian_P.clear();

  SparseMatd ATA, ATB, BTB;
  SparseMatd LML = ldm.U_ext.transpose() * ldm.M * ldm.U_ext;

  const SparseMatd A_ = ldm.MU_ext;
  const SparseMatd B_ = ldm.LU_ext;

  ATA = A_.transpose() * A_;
  ATB = A_.transpose() * B_;
  BTB = B_.transpose() * B_;

  for (const SampleInfo& info : batch.samples) {
    SparseMatd B(n_controls * dim, n_args * dim);
    SparseMatd Bdd(n_controls * dim, n_args * dim);
    SparseMatd Bd(n_controls * dim, n_args * dim);

    int idx = info.keyframe_idx;
    double T = T_between[idx];
    double t = info.t;

    for (int i = 0; i < n_controls; i++) {
      int p1_idx = argPV.keyframe_idx[idx].P_idx_in_arg(i);
      int p2_idx = argPV.keyframe_idx[idx + 1].P_idx_in_arg(i);
      int v1_idx = argPV.keyframe_idx[idx].V_idx_in_arg(i);
      int v2_idx = argPV.keyframe_idx[idx + 1].V_idx_in_arg(i);

      Vec4d coeff = get_bezier_coeff(t);
      Vec4d d_coeff = get_bezier_d_coeff(t) / T;
      Vec4d dd_coeff = get_bezier_dd_coeff(t) / T / T;
      Eigen::VectorXd I = Eigen::VectorXd::Ones(dim);

      if (p1_idx != -1) {
        int arg_idx = p1_idx;
        set_sparse_block_from_diagnol(B, coeff(0) * I, i * dim, arg_idx * dim);
        set_sparse_block_from_diagnol(Bd, d_coeff(0) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(Bdd, dd_coeff(0) * I, i * dim,
                                      arg_idx * dim);
      }
      if (p2_idx != -1) {
        int arg_idx = p2_idx;
        set_sparse_block_from_diagnol(B, coeff(1) * I, i * dim, arg_idx * dim);
        set_sparse_block_from_diagnol(Bd, d_coeff(1) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(Bdd, dd_coeff(1) * I, i * dim,
                                      arg_idx * dim);
      }
      if (v1_idx != -1) {
        int arg_idx = v1_idx;
        set_sparse_block_from_diagnol(B, T * coeff(2) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(Bd, T * d_coeff(2) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(Bdd, T * dd_coeff(2) * I, i * dim,
                                      arg_idx * dim);
      }
      if (v2_idx != -1) {
        int arg_idx = v2_idx;
        set_sparse_block_from_diagnol(B, T * coeff(3) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(Bd, T * d_coeff(3) * I, i * dim,
                                      arg_idx * dim);
        set_sparse_block_from_diagnol(Bdd, T * dd_coeff(3) * I, i * dim,
                                      arg_idx * dim);
      }
    }

    double dtau = info.weight * T_between[info.keyframe_idx];
    SparseMatd tmp1 = ATA * Bdd + ATB * B + damping_alpha * ATA * Bd;
    SparseMatd tmp2 =
        ATB.transpose() * Bdd + BTB * B + damping_alpha * ATB.transpose() * Bd;
    SparseMatd tmp3 = ATA * Bdd + ATB * B + damping_alpha * ATA * Bd;
    SparseMatd hessian = Bdd.transpose() * tmp1 +  //
                         B.transpose() * tmp2 +    //
                         damping_alpha * Bd.transpose() * tmp3;
    SparseMatd J_arg = A_ * Bdd + B_ * B + damping_alpha * A_ * Bd;

    Hessian_P.push_back(J_arg);

    sum_hessian += dtau * hessian;
    sum_hessian += dtau * 1.0 / h / h * (B.transpose() * LML * B);
  }
}

double summarize_lbs_jacobian_PV_with_pre(
    const SampleBatch& batch,                  //
    const ArgIdxPV& argPV,                     //
    const SplineTrajectory& trajectory,        //
    ProjectiveDynamicsSolver<3>& pd_solver,    //
    const LBSDataManager& ldm,                 //
    Vecxd& sum_J_arg,                          //
    const std::vector<SparseMatd>& Hessian_p,  //
    const double damping_alpha) {
  int dim = 3;
  int n_verts = ldm.n_verts;
  const int n_samples = batch.n_samples;
  const int n_controls = ldm.n_controls;
  const int n_args = argPV.n_argP + argPV.n_argV;
  sum_J_arg.setZero();
  double Wsp = 0.0;
  int idx = 0;

  MatxXd c_p(n_controls, dim);
  MatxXd c_v(n_controls, dim);
  MatxXd c_a(n_controls, dim);
  Vecxd material_jacobian(n_verts * 3);

  for (int s = 0; s < batch.n_samples; s++) {
    const SampleInfo& info = batch.samples[s];
    double T = trajectory.T_between[info.keyframe_idx];
    sample_damped_dynamic_status(info, trajectory, c_p, c_v, c_a);
    Vecxd c_v_vec = aphys::Vecxd::Map(c_v.data(), c_v.size());
    Vecxd c_a_vec = aphys::Vecxd::Map(c_a.data(), c_a.size());

    MatxXd pos = ldm.U * c_p;
    pd_solver.localStep(pos, ldm.tets);
    MatxXd gradient = ldm.LU * c_p - pd_solver.J * pd_solver.P;
    material_jacobian = Vecxd::Map(gradient.data(), gradient.size());

    double W = 0.0;

    Vecxd f = material_jacobian + ldm.MU_ext * c_a_vec / T / T +
              damping_alpha * ldm.MU_ext * c_v_vec / T;

    double dtau = info.weight * trajectory.T_between[info.keyframe_idx];
    const SparseMatd& B = Hessian_p[idx];
    Wsp += dtau * 0.5 * f.dot(f);
    sum_J_arg += dtau * B.transpose() * f;
    idx++;
  }
  return Wsp;
}

}  // namespace aphys
