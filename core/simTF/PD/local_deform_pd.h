#ifndef SIMTF_PD_LOCAL_DEFORM_PD_H_
#define SIMTF_PD_LOCAL_DEFORM_PD_H_

#include <ArmorerPhys/type.h>
#include <Eigen/SparseLU>

#include <memory>

#include "simTF/type.h"

namespace aphys {

template <int dim>
struct LocalDeformParams {
  typedef Eigen::Matrix<int, Eigen::Dynamic, dim + 1, Eigen::RowMajor> EleMat;
  const EleMat& elements;
  const Vecxd& element_mass;
  const Vecxd& vert_mass;
  const MatxXd& external_force;
  const double stiffness_hydro;
  const double stiffness_devia;
  const double local_stiffness;

  LocalDeformParams(const EleMat& elements,  //
                    const Vecxd& element_mass,
                    const Vecxd& vert_mass,  //
                    const MatxXd& external_force, const double stiffness_hydro,
                    const double stiffness_devia, const double local_stiffness)
      : elements(elements),
        element_mass(element_mass),
        vert_mass(vert_mass),
        external_force(external_force),
        stiffness_hydro(stiffness_hydro),
        stiffness_devia(stiffness_devia),
        local_stiffness(local_stiffness) {}
};

// Local deform solver with one selected vertex for manipulation
// 1. Initialize this local-global solver when the keyframe is selected
// 2. Only one linear constraint: the selected vertex is fixed
// 3. Use a simple L2 norm to control the locality of the deformation
template <int dim>
struct LocalDeformSolver {
  typedef Eigen::Matrix<int, Eigen::Dynamic, dim + 1, Eigen::RowMajor> EleMat;
  typedef Eigen::Matrix<double, dim, dim> Mat;
  typedef Eigen::Matrix<double, dim, 1> Vec;

  int n_verts;
  int n_elements;
  std::vector<Mat> dx_ref_inv;
  MatxXd P;
  SparseMatd L;
  SparseMatd K;
  SparseMatd J;
  MatxXd LHS;
  MatxXd rhs_KX_ref;
  SparseMatd LHS_sparse;
  double ratio;

  std::unique_ptr<LocalDeformParams<dim>> params;

  Eigen::SparseLU<SparseMatd, Eigen::COLAMDOrdering<int>> sparse_solver;

  LocalDeformSolver() {}

  void initializeParams(const EleMat& elements,  //
                        const Vecxd& element_mass,
                        const Vecxd& vert_mass,  //
                        const MatxXd& external_force,
                        const double stiffness_hydro,
                        const double stiffness_devia,
                        const double local_stiffness);

  void setupDeformer(const MatxXd& verts_ref,  //
                     const int selected_vertex);

  void localStep(const MatxXd& verts);
  void globalStep(MatxXd& verts, Vecxd fixed_pos);
};

}  // namespace aphys

#endif  // SIMTF_PD_LOCAL_DEFORM_PD_H_
