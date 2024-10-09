
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

#include "simTF/spline.h"
#include "simTF/PD/pd.h"
#include "simTF/optimize.h"
#include "simTF/optimize/damping_version.h"
#include "simTF/renderer/renderer.h"
#include "simTF/lbs/model.h"
#include "simTF/export.h"

int main() {
  using namespace aphys;
  // =========================== prepare gl enviromnent
  // ===========================
  const unsigned int SCR_WIDTH = 700;
  const unsigned int SCR_HEIGHT = 700;
  GLFWwindow* window = aphys::create_window(SCR_WIDTH, SCR_HEIGHT,
                                            "Eulerized Dino Interpolation");

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
          {3.0f, 0.5f, 0.5f},                   // camera target
          {0.0f, 1.0f, 0.0f},                   // camera up axis
          float(SCR_WIDTH) / float(SCR_HEIGHT)  // camera aspect
          ));
  aphys::Vec3f diffuse_color(0.0f, 211.f / 255.f, 239.f / 255.f);
  aphys::Gui gui = aphys::create_gui(window, "gui");
  gui.width = SCR_WIDTH;
  gui.height = 80;

  // =========================== prepare data ===========================
  MatxXd v_ref;
  MatxXd v_p, v_acce, v_vel;
  TetMesh tm;
  VisualTetMesh vtm;
  Vecxi is_tail;

  std::string cnslash = "/";
  std::string assets_path = SIMTF_RESOURCES_PATH + cnslash + "dino" + cnslash;
  {
    std::string ref_file = assets_path + "dino.mesh";
    igl::readMESH(ref_file, tm.verts, tm.tets, tm.faces);
    v_ref = tm.verts;
    v_p = v_ref;
    v_acce = v_ref;
    v_acce.setZero();
    extract_visual_tets_surfaces(tm.tets, tm.verts, vtm.visual_faces);
    std::cout << "n_verts: " << tm.verts.rows() << std::endl;
    std::cout << "n_tets: " << tm.tets.rows() << std::endl;
  }
  int n_verts = tm.verts.rows();
  int n_faces = tm.faces.rows();
  int n_tets = tm.tets.rows();

  std::vector<MatxXd> key_frames(10);
  for (int i = 0; i < key_frames.size(); i++) {
    std::string filename =
        assets_path + "dino_tet_" + std::to_string(i) + ".mesh";
    Matx3i F;
    Matx4i T;
    igl::readMESH(filename, key_frames[i], T, F);
  }

  MatxXd V0;
  MatxXi F0;
  igl::readOBJ(assets_path + "dino.obj", V0, F0);
  aphys::SparseMatd bind_mat;
  aphys::MatxXd bc_weights;
  aphys::Vecxi bc_index;
  igl::readDMAT(assets_path + "dino_weight_weight.dmat", bc_weights);
  igl::readDMAT(assets_path + "dino_weight_idx.dmat", bc_index);
  igl::readDMAT(assets_path + "dino_is_tail.dmat", is_tail);

  Timer::getInstance()->start("precompute");

  aphys::generate_bind_mat(tm.verts.rows(), tm.tets, bc_index, bc_weights,
                           bind_mat);

  std::vector<int> segments = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  // std::vector<int> segments = {2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<double> time_interval(9);
  for (double& t : time_interval) {
    t = 0.35;
  }

  double total_time = sum_vector(time_interval);
  std::vector<int> n_samples_interval = {10, 10, 10, 10, 10, 10, 10, 10, 10};
  // std::vector<int> n_samples_interval = {8, 8, 8, 8, 8, 8, 8, 8, 8,
  //                                        8, 8, 8, 8, 8, 8, 8, 8, 8};

  SplineTrajectory trajectory;
  set_keyframe_bezier_interpolation(key_frames, segments, time_interval,
                                    trajectory);

  Vecxd tet_mass, vert_mass;
  compute_tet_mass(tm.verts, tm.tets, tet_mass, vert_mass, 1.0);
  DiagMatxXd v_M;
  set_diag_matrix(vert_mass, v_M, 3);
  DiagMatxXd v_InvM = v_M.inverse();

  MatxXd ext_force(n_verts, 3);
  ext_force.col(1).array() = -0.0;
  double devia_stiffness = 60.0f;
  double hydro_stiffness = 30.0f;
  Box3d box(-1.0, 1.0, -2.0, 3.0, -1.0, 1.0);

  double dt = 1.0 / 60.0;
  double damping_alpha = 0.0;
  ProjectiveDynamicsSolver<3> pd_solver(v_p, v_ref, tm.tets, tet_mass,
                                        vert_mass, ext_force, dt,
                                        hydro_stiffness, devia_stiffness);

  SampleBatch sample_batch;
  build_sample_batch(n_samples_interval, sample_batch, trajectory);

  // =========================== set fixed incides ===========================
  for (int i = 0; i < trajectory.n_keyframes; i++) {
    if (i == 0 || i == trajectory.n_keyframes - 1) {
      trajectory.keyframes[i].p_fix.fill(1);
    } else {
      for (int j = 0; j < n_verts; j++) {
        if (is_tail(j) == 1)
          trajectory.keyframes[i].p_fix(j) = 0;
        else
          trajectory.keyframes[i].p_fix(j) = 1;
      }
    }
  }

  ArgIdxPV arg_pv;
  initialize_argPV(trajectory, arg_pv);
  int n_args = arg_pv.n_argP + arg_pv.n_argV;
  Vecxd arg(n_args * 3);
  Vecxd arg_solver(n_args * 3);
  select_optimize_argPV<3>(trajectory, arg_pv, arg);

  // =========================== optimize ===========================
  Vec3d gravity(0.0, -4.0, 0.0);
  DataManager<3> data(vert_mass, damping_alpha, tm.tets, gravity);
  ArgSelection<3> arg_selection(trajectory, arg_pv);
  BezierMatrix<3> optimizer(trajectory, arg_pv, sample_batch, pd_solver,
                            arg_selection, data);
  auto optimize_trajectory_PV = [&]() {
    std::cout << compute_damped_Wsp<3>(sample_batch, vert_mass, trajectory,
                                       pd_solver.energy_jacobian_func,
                                       damping_alpha)
              << std::endl;

    Timer::getInstance()->start("PV optimize");
    arg_selection.update_selection();
    optimizer.local_step();
    optimizer.global_step();
    Timer::getInstance()->pause("PV optimize");
    Timer::getInstance()->print_all();
    print_tet_info(tm);
    std::cout << compute_damped_Wsp<3>(sample_batch, vert_mass, trajectory,
                                       pd_solver.energy_jacobian_func,
                                       damping_alpha)
              << std::endl;
  };

  // =========================== implicit-eulerized optimization

  KeyframeEditingInterface keyframe_interface(scene, gui, trajectory, tm,
                                              material);
  keyframe_interface.setup_deformation_params(tm.tets, tet_mass, vert_mass,
                                              ext_force, hydro_stiffness,
                                              devia_stiffness, 0.03);

  Timer::getInstance()->pause("precompute");

  DiffuseMesh origin_mesh = create_diffuse_mesh(material);
  add_render_func(scene, get_render_func(origin_mesh));
  DiffuseMesh wire_mesh = create_diffuse_mesh(wire_material);
  add_render_func(scene, get_render_func(wire_mesh), true, true);

  double prev_loss = 0.0;

  float start_time = glfwGetTime();
  float tau = total_time;
  std::string output_path =
      SIMTF_RESOURCES_PATH + cnslash + "../output" + cnslash;
  char filename[128] = "";

  add_gui_func(gui, [&]() {
    bool pressed = ImGui::Button("Optimize");
    if (pressed) {
      // reg.regularize(trajectory, sum_HR, rhs, arg_pv, new_arg);
      // apply_arg_in_solver(arg_pv, traj_solver, new_arg, trajectory);
      optimize_trajectory_PV();
    }
    bool pressed2 = ImGui::Button("Optimize T");
    if (pressed2) {
      // optimize_trajectory_T();
    }
    bool play = ImGui::Button("Play");
    if (play) {
      start_time = glfwGetTime();
      tau = 0.0;
    }
    ImGui::InputText("filename", filename, 128);

    bool export_usd = ImGui::Button("Save");
    if (export_usd) {
      // TrajectoryExporter exporter(trajectory, tm.tets, 20);
      TrajectoryExporter exporter(trajectory, F0, bind_mat, 10);
      exporter.SaveAnimationToUSD(output_path + filename + ".usdc");

      SaveFrameAnimatinoToUsd(trajectory, F0, bind_mat,
                              output_path + filename + "_frame.usdc", 60);
    }
  });

  glfwSwapInterval(1);

  while (!glfwWindowShouldClose(window)) {
    tau = (glfwGetTime() - start_time) / (3.0 / total_time);
    glfwPollEvents();
    handle_gui_input(gui);

    if (tau < total_time) {
      int idx =
          sample_damped_tau_on_trajectory(tau, trajectory, v_p, v_vel, v_acce);
      construct_visual_tets(vtm.visual_verts, v_p, tm.tets);
      set_mesh_data(origin_mesh, vtm.visual_verts.cast<float>(),
                    vtm.visual_faces);
      // V0 = bind_mat * v_p;
      // set_mesh_data(origin_mesh, V0.cast<float>(), F0);
      // set_mesh_data(wire_mesh, V0.cast<float>(), F0);
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
