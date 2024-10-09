#ifndef SIMTF_OPTIMIZE_DAMPING_VERSION_H_
#define SIMTF_OPTIMIZE_DAMPING_VERSION_H_

#include <ArmorerPhys/type.h>

#include "simTF/spline.h"
#include "simTF/type.h"
#include "simTF/optimize/newton.h"

namespace aphys {

void sample_damped_dynamic_status(  //
    const SampleInfo& sample_info,
    const SplineTrajectory& trajectory,         //
    MatxXd& v_p, MatxXd& v_vel, MatxXd& v_acce  //
);

template <int dim>
void compute_damped_force_residual(Vecxd& f, const Vecxd& material_jacobian,
                                   const MatxXd& v_vel, const MatxXd& v_acce,
                                   const Vecxd& vert_mass, const double currT,
                                   const double damping_alpha);

template <int dim>
double compute_damped_Wsp(const SampleBatch& batch,            //
                          const Vecxd& vert_mass,              //
                          const SplineTrajectory& trajectory,  //
                          const Energy_Jacobian_Func& energy_jacobian_func,
                          const double damping_alpha);

//! include energy dissipation term damping
template <int dim>
void precompute_damped_eulerized_hessian_batch(
    const SampleBatch& batch, const ArgIdxPV& argPV,
    const std::vector<double> T_between, const Vecxd& vert_mass,
    const SparseMatd& L_ext, const double damping_alpha,
    std::vector<SparseMatd>& hessians, std::vector<SparseMatd>& hessians_euler,
    SparseMatd& sum_hessian, double h);

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
    const double damping_alpha);

int sample_damped_tau_on_trajectory(const double tau,              //
                                    SplineTrajectory& trajectory,  //
                                    MatxXd& pos, MatxXd& vel, MatxXd& acce);

template <int dim>
void summarize_damped_derivative_T(
    const SampleBatch& batch,            //
    const SplineTrajectory& trajectory,  //
    const Vecxd& vert_mass,              //
    const Energy_Jacobian_Func& energy_jacobian_func, const SparseMatd& L_ext,
    Vecxd& sum_J_T, Vecxd& sum_H_T, const double damping_alpha);

template <int dim>
void T_damped_line_search(const SampleBatch& batch,      //
                          SplineTrajectory& trajectory,  //
                          const Vecxd& gradient_T_direc,
                          const Vecxd& search_T_direc, const Vecxd& vert_mass,
                          const Energy_Jacobian_Func& energy_jacobian_func,
                          const double damping_alpha);

}  // namespace aphys

#endif  // SIMTF_OPTIMIZE_DAMPING_VERSION_H_
