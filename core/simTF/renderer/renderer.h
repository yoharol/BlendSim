#ifndef SIMTF_RENDERER_RENDERER_H_
#define SIMTF_RENDERER_RENDERER_H_

#include <ArmorerPhys/type.h>
#include <ArmorerPhys/RenderCore.h>

#include <vector>

#include "simTF/type.h"
#include "simTF/spline.h"
#include "simTF/PD/local_deform_pd.h"

namespace aphys {

struct KeyframeEditingInterface {
  DiffuseMesh keyframe_mesh;
  std::vector<MatxXd> visual_verts;
  Matx3i visual_faces;
  Matx3i surface_faces;
  int curr_kdx = 0;
  bool editing_keyframe = false;
  bool dragging = false;
  int edit_idx = -1;
  Vec3d start_edit_pos;
  Vec3d start_edit_ref_pos;
  double ray_dis;

  SplineTrajectory& traj;
  const Matx4i& tet_faces;

  LocalDeformSolver<3> local_deform_solver;

  KeyframeEditingInterface(Scene& scene, Gui& gui, SplineTrajectory& trajectory,
                           const TetMesh& tet_mesh, DiffuseMaterial& material);

  void reset_mesh(MatxXd& v_p);

  void set_selection(int idx, float alpha = 1.0f);

  void setup_editing(int idx, bool editing);

  void setup_deformation_params(const Matx4i& elements,  //
                                const Vecxd& element_mass,
                                const Vecxd& vert_mass,  //
                                const MatxXd& external_force,
                                const double stiffness_hydro,
                                const double stiffness_devia,
                                const double local_stiffness);

  void reset_keyframes(SplineTrajectory& trajectory);
};

/*struct KeyframeEditingInterface2D {
  Edges keyframe_edges;
  const Matx3i& faces;
  const Matx2i& edges;
  int curr_kdx = 0;
  bool editing_keyframe = false;
  bool dragging = false;
  int edit_idx = -1;
  Vec2d start_edit_pos;
  Vec2d start_edit_ref_pos;

  SplineTrajectory& traj;

  LocalDeformSolver<2> local_deform_solver;

  KeyframeEditingInterface2D(Scene& scene, Gui& gui,
                             SplineTrajectory& trajectory, const Matx3i& faces,
                             const Matx2i& edges, DiffuseMaterial& material);

  void reset_mesh(MatxXd& v_p);

  void set_selection(int idx, float alpha = 1.0f);

  void setup_editing(int idx, bool editing);

  void setup_deformation_params(const Matx3i& faces,  //
                                const Vecxd& element_mass,
                                const Vecxd& vert_mass,  //
                                const MatxXd& external_force,
                                const double stiffness_hydro,
                                const double stiffness_devia,
                                const double local_stiffness);
};*/

}  // namespace aphys

#endif  // SIMTF_RENDERER_RENDERER_H_
