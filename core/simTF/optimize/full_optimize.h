#ifndef SIMTF_OPTIMIZE_FULL_OPTIMIZE_H_
#define SIMTF_OPTIMIZE_FULL_OPTIMIZE_H_

#include <ArmorerPhys/type.h>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#include "simTF/PD/pd.h"
#include "simTF/type.h"
#include "simTF/spline.h"
#include "simTF/lbs/model.h"

namespace aphys {

// A lazy dummy for me to pass the varialbes
template <int dim>
struct DataManager {
  typedef Eigen::Matrix<int, Eigen::Dynamic, dim + 1, Eigen::RowMajor> EleMat;

  Vecxd& vert_mass;
  SparseMatd M;
  SparseMatd M_ext;
  double damping_alpha;
  EleMat elements;
  MatxXd ext_F;

  DataManager(Vecxd& vert_mass, double damping_alpha, EleMat& elements,
              Vecxd ext_force);
};

template <int dim>
struct ArgSelection {
  SplineTrajectory& trajectory;
  ArgIdxPV& arg_pv;
  SparseMatd S;
  MatxXd C;

  ArgSelection(SplineTrajectory& trajectory, ArgIdxPV& arg_pv);

  void update_selection();
};

template <int dim>
struct BezierMatrix {
  int n_verts;
  int n_keyframes;
  int n_fullargs;
  int n_args;

  std::vector<SparseMatd> JacobianFres;
  std::vector<SparseMatd> Jacobian2;
  SparseMatd sum_lhs_hessian;
  SparseMatd sum_rhs_hessian;
  Eigen::MatrixXd rhs;
  Eigen::MatrixXd test_rhs;
  ArgSelection<dim>& arg_selection;
  Eigen::SimplicialLDLT<SparseMatd> solver;

  SplineTrajectory& trajectory;
  ArgIdxPV& arg_pv;
  SampleBatch& sample_batch;
  ProjectiveDynamicsSolver<dim>& pd_solver;
  DataManager<dim>& data;

  BezierMatrix(SplineTrajectory& trajectory, ArgIdxPV& arg_pv,
               SampleBatch& sample_batch,
               ProjectiveDynamicsSolver<dim>& pd_solver,
               ArgSelection<dim>& arg_selection, DataManager<dim>& data);

  void ReCompute();

  void local_step();

  void global_step();
};

template <int dim>
struct BezierLBSMatrix {
  int n_verts;
  int n_controls;
  int n_keyframes;
  int n_fullargs;
  int n_args;

  std::vector<SparseMatd> JacobianFres;
  std::vector<SparseMatd> Jacobian2;
  SparseMatd sum_lhs_hessian;
  SparseMatd sum_rhs_hessian;
  Eigen::MatrixXd rhs;
  Eigen::MatrixXd test_rhs;
  ArgSelection<dim>& arg_selection;
  Eigen::SimplicialLDLT<SparseMatd> solver;

  LBSModel& lbs_model;
  SplineTrajectory& trajectory;
  ArgIdxPV& arg_pv;
  SampleBatch& sample_batch;
  ProjectiveDynamicsSolver<dim>& pd_solver;
  DataManager<dim>& data;

  BezierLBSMatrix(SplineTrajectory& trajectory, ArgIdxPV& arg_pv,
                  SampleBatch& sample_batch,
                  ProjectiveDynamicsSolver<dim>& pd_solver,
                  ArgSelection<dim>& arg_selection, DataManager<dim>& data,
                  LBSModel& lbs_model);

  void local_step();

  void global_step();
};

}  // namespace aphys

#endif  // SIMTF_OPTIMIZE_FULL_OPTIMIZE_H_
