#ifndef SIMTF_OPTIMIZE_NEWTON_OPTIMIZE_H_
#define SIMTF_OPTIMIZE_NEWTON_OPTIMIZE_H_

#include <ArmorerPhys/type.h>

#include "simTF/spline.h"
#include "simTF/type.h"

namespace aphys {

template <int dim>
void compute_force_residual(Vecxd& f, const Vecxd& material_jacobian,
                            const MatxXd& v_acce, const Vecxd& vert_mass,
                            const double currT);

template <int dim>
double compute_Wsp(const SampleBatch& batch,            //
                   const Vecxd& vert_mass,              //
                   const SplineTrajectory& trajectory,  //
                   const Energy_Jacobian_Func& energy_jacobian_func);

template <int dim>
void precompute_hessian_batch(const SampleBatch& batch, const ArgIdxPV& argPV,
                              const std::vector<double> T_between,
                              const Vecxd& vert_mass, const SparseMatd& L_ext,
                              std::vector<SparseMatd>& hessians,
                              SparseMatd& sum_hessian);

template <int dim>
void precompute_eulerized_hessian_batch(
    const SampleBatch& batch, const ArgIdxPV& argPV,
    const std::vector<double> T_between, const Vecxd& vert_mass,
    const SparseMatd& L_ext, std::vector<SparseMatd>& hessians,
    std::vector<SparseMatd>& hessians_euler, SparseMatd& sum_hessian, double h);

template <int dim>
void summarize_derivative_T(const SampleBatch& batch,            //
                            const SplineTrajectory& trajectory,  //
                            const Vecxd& vert_mass,              //
                            const Energy_Jacobian_Func& energy_jacobian_func,
                            const SparseMatd& L_ext, Vecxd& sum_J_T,
                            Vecxd& sum_H_T);

template <int dim>
double summarize_derivative_PV_with_pre(
    const SampleBatch& batch,                         //
    const ArgIdxPV& argPV,                            //
    const SplineTrajectory& trajectory,               //
    const Vecxd& vert_mass,                           //
    Vecxd& sum_J_arg,                                 //
    std::vector<SparseMatd>& hessians,                //
    const Energy_Jacobian_Func& energy_jacobian_func  //
);

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
    const SplineTrajectory& pred_trajectory);

template <int dim>
void compute_arg_jacobian_T(const SampleInfo& info,              //
                            const ArgIdxPV& argPV,               //
                            const SplineTrajectory& trajectory,  //
                            const Vecxd& vert_mass,              //
                            const SparseMatd& dFdx,              //
                            SparseMatd& sample_J_arg);

template <int dim>
void line_search(const SampleBatch& batch,  //
                 Vecxd& curr_arg, Vecxd& search_direc,
                 Vecxd& arg_solver,                                //
                 const Vecxd& J,                                   //
                 Vecxd& energy_jacobian,                           //
                 Vecxd& f,                                         //
                 const Vecxd& vert_mass,                           //
                 const DiagMatxXd& M_inv,                          //
                 Vecxd& mass_tilde,                                //
                 const SplineTrajectory& trajectory,               //
                 const ArgIdxPV& argPV,                            //
                 SplineTrajectory& traj_solver,                    //
                 const Energy_Jacobian_Func& energy_jacobian_func  //
);

template <int dim>
void apply_arg_in_solver(const ArgIdxPV& argPV,               //
                         const SplineTrajectory& trajectory,  //
                         const Vecxd& curr_arg, SplineTrajectory& traj_new);

void apply_arg_in_solver(const ArgIdxPV& argPV,               //
                         const SplineTrajectory& trajectory,  //
                         const MatxXd& curr_arg, SplineTrajectory& traj_new);

template <int dim>
void T_line_search(const SampleBatch& batch,      //
                   SplineTrajectory& trajectory,  //
                   const Vecxd& gradient_T_direc, const Vecxd& search_T_direc,
                   const Vecxd& vert_mass,
                   const Energy_Jacobian_Func& energy_jacobian_func);

}  // namespace aphys

#endif  // STSP_OPTIMIZE_NEWTON_OPTIMIZE_H_
