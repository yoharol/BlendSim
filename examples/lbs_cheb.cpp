#include <vector>

#include "Eigen/SparseQR"
#include "Eigen/Sparse"
#include "Eigen/IterativeLinearSolvers"
#include "Eigen/SparseCholesky"

#include "ArmorerPhys/RenderCore.h"
#include "ArmorerPhys/SimCore.h"
#include "ArmorerPhys/timer.h"
#include "ArmorerPhys/data/tetrahetralize.h"
#include "ArmorerPhys/data/voxel_builder.h"
#include "igl/readOBJ.h"
#include "igl/readMESH.h"
#include "igl/readDMAT.h"
#include "igl/writeDMAT.h"
#include "igl/lbs_matrix.h"

#include "simTF/spline.h"
#include "simTF/PD/pd.h"
#include "simTF/optimize.h"
#include "simTF/renderer/renderer.h"
#include "simTF/lbs/model.h"

int main() {
  using namespace aphys;
  // =========================== prepare gl enviromnent
  // ===========================
  const unsigned int SCR_WIDTH = 800;
  const unsigned int SCR_HEIGHT = 800;
  GLFWwindow* window =
      aphys::create_window(SCR_WIDTH, SCR_HEIGHT, "Cheb Viewer");

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
          {6.0f, 0.35f, -2.0f}  // light position
      },
      aphys::create_camera(
          {6.44f, 0.33f, -1.57f},               // camera position
          {0.0f, 0.5f, 0.0f},                   // camera target
          {0.0f, 1.0f, 0.0f},                   // camera up axis
          float(SCR_WIDTH) / float(SCR_HEIGHT)  // camera aspect
          ));
  aphys::Vec3f diffuse_color(0.0f, 211.f / 255.f, 239.f / 255.f);
  aphys::Gui gui = aphys::create_gui(window, "gui");
  gui.width = 300;
  gui.height = 200;

  // =========================== prepare data ===========================

  TetMesh tm;
  VisualTetMesh vtm;
  Matx3f tm_color;

  std::string cnslash = "/";
  std::string assets_path = SIMTF_RESOURCES_PATH + cnslash + "cheb" + cnslash;

  igl::readMESH(assets_path + "cheb.mesh", tm.verts, tm.tets, tm.faces);

  extract_visual_tets_surfaces(tm.tets, tm.verts, vtm.visual_faces);
  // tm_color.resize(tm.verts.rows(), 3);
  // tm_color.setConstant(0.5f);

  LBSModel lbs_model(assets_path + "cheb_ref_points.dmat",  //
                     "",                                    //
                     assets_path + "cheb_bone_edges.dmat",  //
                     assets_path + "cheb_bbw_weight.dmat",  //
                     assets_path + "cheb_c_weight.dmat");

  lbs_model.load_anim_data(assets_path + "cheb_selected_anim.dmat");
  lbs_model.initialize_lbs_weights(tm.verts);

  int n_frames = lbs_model.anim_frame_data.rows();
  SplineTrajectory lbs_traj;
  double total_time;
  SampleBatch sample_batch;
  {
    std::vector<MatxXd> key_frames(n_frames);
    for (int k = 0; k < n_frames; k++) {
      lbs_model.load_frame_data(k);
      key_frames[k] = lbs_model.T;
    }
    std::vector<int> segments(n_frames - 1);
    for (int i = 0; i < n_frames - 1; i++) {
      segments[i] = 1;
    }
    std::vector<double> time_interval(n_frames - 1);
    for (double& t : time_interval) {
      t = 0.25;
    }
    total_time = sum_vector(time_interval);
    set_keyframe_bezier_interpolation(key_frames, segments, time_interval,
                                      lbs_traj);

    std::vector<int> n_samples_interval(n_frames - 1);
    for (int& n : n_samples_interval) {
      n = 10;
    }
    build_sample_batch(n_samples_interval, sample_batch, lbs_traj);
  }
  int n_controls = lbs_traj.n_verts;

  MatxXd v_p = tm.verts;
  Vecxd tet_mass, vert_mass;
  int n_verts = tm.verts.rows();
  compute_tet_mass(tm.verts, tm.tets, tet_mass, vert_mass, 100.0);
  DiagMatxXd M_ext(n_verts * 3);
  set_diag_matrix(vert_mass, M_ext, 3);

  MatxXd ext = tm.verts;
  ext.setZero();
  double devia_stiffness = 1.0f;
  double hydro_stiffness = 1.0f;
  double dt = 1.0 / 360.0;
  double damping_alpha = 0.0;
  Timer::getInstance()->start("pd_solver");
  ProjectiveDynamicsSolver<3> pd_solver(v_p, tm.verts, tm.tets, tet_mass,
                                        vert_mass, ext, dt, hydro_stiffness,
                                        devia_stiffness);

  LBSDataManager ldm(n_controls, pd_solver, lbs_model, tm, M_ext);

  double w0 = compute_lbs_wsp(sample_batch, n_verts, lbs_traj, pd_solver, ldm,
                              damping_alpha);
  Timer::getInstance()->pause("pd_solver");
  // exit(-1);
  std::cout << compute_lbs_wsp(sample_batch, n_verts, lbs_traj, pd_solver, ldm,
                               damping_alpha)
            << "\n";

  // =========================== set fixed incides ===========================
  for (int i = 0; i < lbs_traj.n_keyframes; i++) {
    lbs_traj.keyframes[i].p_fix.fill(1);
  }
  lbs_traj.keyframes[0].v_fix.fill(1);
  lbs_traj.keyframes[lbs_traj.n_keyframes - 1].v_fix.fill(1);

  std::cout << compute_lbs_wsp(sample_batch, n_verts, lbs_traj, pd_solver, ldm,
                               damping_alpha)
            << "\n";

  Timer::getInstance()->start("summarize arg");
  ArgIdxPV arg_pv;
  initialize_argPV(lbs_traj, arg_pv);
  int n_args = arg_pv.n_argP + arg_pv.n_argV;
  Vecxd arg(n_args * 3);
  Vecxd arg_solver(n_args * 3);
  Vecxd arg_J(n_args * 3);
  select_optimize_argPV<3>(lbs_traj, arg_pv, arg);
  Timer::getInstance()->pause("summarize arg");

  Timer::getInstance()->start("precompute hessian");
  Vec3d gravity(0.0, 0.0, 0.0);
  DataManager<3> data(vert_mass, damping_alpha, tm.tets, gravity);
  ArgSelection<3> arg_selection(lbs_traj, arg_pv);
  BezierLBSMatrix<3> optimizer(lbs_traj, arg_pv, sample_batch, pd_solver,
                               arg_selection, data, lbs_model);
  Timer::getInstance()->pause("precompute hessian");

  Timer::getInstance()->print_all();

  construct_visual_tets(vtm.visual_verts, tm.verts, tm.tets);
  DiffuseMesh traj_mesh = create_diffuse_mesh(material);
  add_render_func(scene, get_render_func(traj_mesh), true, true);
  lbs_model.initialize_render_objects(scene);

  double prev_loss = 0.0;

  auto optimize_trajectory_PV = [&]() {
    optimizer.local_step();
    optimizer.global_step();
  };

  add_gui_func(gui, [&]() {
    if (ImGui::Button("Optimize")) {
      optimize_trajectory_PV();
    }
  });

  glfwSwapInterval(1);

  double start_time = glfwGetTime();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    handle_gui_input(gui);

    double tau = (glfwGetTime() - start_time) / (4.0 / total_time);
    if (tau > total_time) {
      tau = 0.0;
      start_time = glfwGetTime();
    }

    aphys::MatxXd sampled_T;
    aphys::MatxXd sampled_T_acce;
    aphys::MatxXd verts;
    sample_tau_on_trajectory(tau, lbs_traj, sampled_T, sampled_T_acce);
    verts = ldm.U * sampled_T;
    lbs_model.load_frame_affine(sampled_T);

    construct_visual_tets(vtm.visual_verts, verts, tm.tets);
    set_mesh_data(traj_mesh, vtm.visual_verts.cast<float>(), vtm.visual_faces);

    aphys::set_background_RGB({244, 244, 244});
    aphys::orbit_camera_control(window, scene.camera, 10.0, scene.delta_time);
    render_scene(scene);
    render_gui(gui);
    glfwSwapBuffers(window);
  }
  destroy_gui(gui);
  glfwTerminate();
}
