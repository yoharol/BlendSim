#include <vector>

#include "Eigen/SparseQR"
#include "Eigen/Sparse"
#include "Eigen/IterativeLinearSolvers"
#include "Eigen/SparseCholesky"

#include "ArmorerPhys/RenderCore.h"
#include "ArmorerPhys/SimCore.h"
#include "ArmorerPhys/timer.h"
#include "igl/readMESH.h"
#include "igl/readDMAT.h"

#include "simTF/spline.h"
#include "simTF/PD/pd.h"
#include "simTF/optimize.h"
#include "simTF/optimize/damping_version.h"
#include "simTF/renderer/renderer.h"
#include "simTF/export.h"

int main() {
  using namespace aphys;
  const unsigned int SCR_WIDTH = 800;
  const unsigned int SCR_HEIGHT = 800;
  GLFWwindow* window = aphys::create_window(SCR_WIDTH, SCR_HEIGHT, "Stretch3D");
  aphys::DiffuseMaterial material{
      {90, 178, 255},   // diffuse color
      {202, 244, 255},  // specular color
      0.0f              // specular strength
  };
  aphys::DiffuseMaterial wire_material{
      {0, 0, 0},  // diffuse color
      {0, 0, 0},  // specular color
      0.5f        // specular strength
  };
  aphys::Scene scene = aphys::create_scene(
      aphys::Light{
          {242, 242, 242},      // light color
          {109, 105, 188},      // ambient color
          {2.0f, 0.35f, -5.0f}  // light position
      },
      aphys::create_camera(
          {-3.0f, 0.0f, -9.0f},                 // camera position
          {0.0f, 0.5f, 0.0f},                   // camera target
          {0.0f, 1.0f, 0.0f},                   // camera up axis
          float(SCR_WIDTH) / float(SCR_HEIGHT)  // camera aspect
          ));
  aphys::Vec3f diffuse_color(0.0f, 211.f / 255.f, 239.f / 255.f);
  aphys::Gui gui = aphys::create_gui(window, "gui");
  gui.width = 300;
  gui.height = 200;

  MatxXd v_ref;
  MatxXd v_p, v_acce, v_vel;
  TetMesh tm;
  VisualTetMesh vtm;

  std::string cnslash = "/";
  std::string assets_path =
      SIMTF_RESOURCES_PATH + cnslash + "stretch3D" + cnslash;
  std::string output_path =
      SIMTF_RESOURCES_PATH + cnslash + "../output" + cnslash;
  {
    std::string ref_file = assets_path + "ref.mesh";
    igl::readMESH(ref_file, tm.verts, tm.tets, tm.faces);
    v_ref = tm.verts;
    v_p = v_ref;
    extract_visual_tets_surfaces(tm.tets, tm.verts, vtm.visual_faces);
    std::cout << "n_verts: " << tm.verts.rows() << std::endl;
    std::cout << "n_tets: " << tm.tets.rows() << std::endl;
  }
  int n_verts = tm.verts.rows();
  int n_faces = tm.faces.rows();
  int n_tets = tm.tets.rows();

  Timer::getInstance()->start("precompute");
  Vecxi top_face_indices;
  get_axis_value_indices(3.0, 1, tm.verts, top_face_indices);
  Vecxi bottom_face_indices;
  get_axis_value_indices(0.0, 1, tm.verts, bottom_face_indices);
  aphys::MatxXf vert_color(v_ref.rows(), 3);
  vert_color.rowwise() =
      aphys::RowVec3f(90.f / 255.f, 178.f / 255.f, 255.f / 255.f);
  for (int i = 0; i < top_face_indices.size(); i++) {
    vert_color.row(top_face_indices(i)) =
        aphys::RowVec3f(196.f / 255.f, 12.f / 255.f, 12.f / 255.f);
  }
  for (int i = 0; i < bottom_face_indices.size(); i++) {
    vert_color.row(bottom_face_indices(i)) =
        aphys::RowVec3f(196.f / 255.f, 12.f / 255.f, 12.f / 255.f);
  }
  aphys::construct_visual_tets_color(vtm.visual_colors, vert_color, tm.tets);

  std::vector<MatxXd> key_frames(3);
  for (int i = 0; i < 3; i++) {
    std::string filename = assets_path + std::to_string(i) + ".mesh";
    Matx3i F;
    Matx4i T;
    igl::readMESH(filename, key_frames[i], T, F);
  }
  key_frames[0] = key_frames[2];

  std::vector<int> segments = {1, 1};
  std::vector<double> time_interval = {0.5, 0.5};
  std::vector<int> n_samples_interval = {7, 7};

  SplineTrajectory trajectory;
  set_keyframe_bezier_interpolation(key_frames, segments, time_interval,
                                    trajectory);

  SampleBatch sample_batch;
  build_sample_batch(n_samples_interval, sample_batch, trajectory);

  Vecxd tet_mass, vert_mass;
  compute_tet_mass(tm.verts, tm.tets, tet_mass, vert_mass);
  DiagMatxXd v_M;
  set_diag_matrix(vert_mass, v_M, 3);
  DiagMatxXd v_InvM = v_M.inverse();

  Vec3d gravity(0.0, 0.0, 0.0);
  MatxXd ext_force;
  generate_gravity_force(gravity, vert_mass, ext_force);
  double devia_stiffness = 30.0f;
  double hydro_stiffness = 30.0f;
  double dt = 1.0 / 60.0;
  double damping_alpha = 1.0;

  ProjectiveDynamicsSolver<3> pd_solver(v_p, v_ref, tm.tets, tet_mass,
                                        vert_mass, ext_force, dt,
                                        hydro_stiffness, devia_stiffness);

  // =========================== set fixed incides ===========================

  trajectory.keyframes[0].p_fix.fill(1);
  trajectory.keyframes[1].p_fix.fill(1);
  trajectory.keyframes[2].p_fix.fill(1);

  ArgIdxPV arg_pv;
  initialize_argPV(trajectory, arg_pv);
  int n_args = arg_pv.n_argP + arg_pv.n_argV;
  Vecxd arg(n_args * 3);
  Vecxd arg_solver(n_args * 3);
  select_optimize_argPV<3>(trajectory, arg_pv, arg);

  Regularizer<3> reg(n_verts);
  SparseMatd sum_HR;
  reg.compute_regularize_gradient(trajectory, sum_HR);
  MatxXd rhs(n_verts * trajectory.n_keyframes * 2, 3);
  rhs.setZero();
  MatxXd new_arg;
  reg.regularize(trajectory, sum_HR, rhs, arg_pv, new_arg);
  apply_arg_in_solver(arg_pv, trajectory, new_arg, trajectory);

  // =========================== regularization ===========================

  Timer::getInstance()->pause("precompute");

  double total_time = sum_vector(trajectory.T_between);
  float start_time = glfwGetTime();
  float tau = total_time;
  bool playing = false;

  ColorMesh color_mesh = create_color_mesh(material);
  add_render_func(scene, get_render_func(color_mesh));

  glfwSwapInterval(1);

  while (!glfwWindowShouldClose(window)) {
    tau = (glfwGetTime() - start_time);
    glfwPollEvents();
    handle_gui_input(gui);

    if (tau < total_time) {
      int idx =
          sample_damped_tau_on_trajectory(tau, trajectory, v_p, v_vel, v_acce);
      construct_visual_tets(vtm.visual_verts, v_p, tm.tets, 0.8);
      set_color_mesh_data(color_mesh, vtm.visual_verts.cast<float>(),
                          vtm.visual_faces, vtm.visual_colors);
    } else {
      start_time = glfwGetTime();
      tau = 0.0;
    }

    aphys::set_background_RGB({244, 244, 244});
    aphys::orbit_camera_control(window, scene.camera, 10.0, scene.delta_time);
    render_scene(scene);
    render_gui(gui);
    glfwSwapBuffers(window);
  }
  destroy_gui(gui);
  glfwTerminate();
}