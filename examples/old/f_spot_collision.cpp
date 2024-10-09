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
#include "simTF/export.h"

int main() {
  using namespace aphys;
  // =========================== prepare gl enviromnent
  // ===========================
  const unsigned int SCR_WIDTH = 800;
  const unsigned int SCR_HEIGHT = 800;
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
          {12.88f, 3.0f, -14.57f},              // camera position
          {6.0f, 3.0f, 0.0f},                   // camera target
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
  std::string assets_path = SIMTF_RESOURCES_PATH + cnslash + "spot" + cnslash;
  {
    std::string ref_file = assets_path + "spot.mesh";
    igl::readMESH(ref_file, tm.verts, tm.tets, tm.faces);
    tm.verts *= 1.6;
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
  Box3d box(-1.0, 13.0, 0.0, 7.0, -6.0, 6.0);

  std::vector<MatxXd> key_frames(4);
  key_frames[0] = tm.verts;
  key_frames[0].col(1).array() += 6.0;
  key_frames[1] = tm.verts;
  key_frames[1].col(0).array() += 6.0;
  key_frames[1].col(1).array() += 0.8;
  key_frames[2] = key_frames[1];
  key_frames[3] = tm.verts;
  key_frames[3].col(0).array() += 12.0;
  key_frames[3].col(1).array() += 4.0;

  // rotate key_frame[3] by 90 degree around z axis
  Eigen::AngleAxisd rot(-M_PI / 4, Eigen::Vector3d::UnitZ());
  Vecxd mean = key_frames[1].colwise().mean();
  for (int i = 0; i < key_frames[2].rows(); i++) {
    key_frames[1].row(i) =
        (rot * (key_frames[1].row(i).transpose() - mean) + mean).transpose();
    key_frames[2].row(i) =
        (rot * (key_frames[2].row(i).transpose() - mean) + mean).transpose();
  }

  Eigen::AngleAxisd rot2(-M_PI / 2, Eigen::Vector3d::UnitZ());
  mean = key_frames[3].colwise().mean();
  for (int i = 0; i < key_frames[3].rows(); i++) {
    key_frames[3].row(i) =
        (rot2 * (key_frames[3].row(i).transpose() - mean) + mean).transpose();
  }

  std::vector<int> segments = {1, 1, 1};
  std::vector<double> time_interval(3);
  for (double& t : time_interval) {
    t = 0.5;
  }
  time_interval[1] = 0.0;
  double total_time = sum_vector(time_interval);
  std::vector<int> n_samples_interval = {12, 0, 12};

  SplineTrajectory trajectory;
  set_keyframe_bezier_interpolation(key_frames, segments, time_interval,
                                    trajectory);

  Vecxd tet_mass, vert_mass;
  compute_tet_mass(tm.verts, tm.tets, tet_mass, vert_mass);
  DiagMatxXd v_M;
  set_diag_matrix(vert_mass, v_M, 3);
  DiagMatxXd v_InvM = v_M.inverse();

  MatxXd ext_force(n_verts, 3);
  ext_force.col(1).array() = -2.0;
  double devia_stiffness = 1.0f;
  double hydro_stiffness = 1.0f;

  double dt = 1.0 / 120.0;
  double damping_alpha = 0.0;
  ProjectiveDynamicsSolver<3> pd_solver(v_p, v_ref, tm.tets, tet_mass,
                                        vert_mass, ext_force, dt,
                                        hydro_stiffness, devia_stiffness);
  SampleBatch sample_batch;
  build_sample_batch(n_samples_interval, sample_batch, trajectory);

  // =========================== set fixed incides ===========================
  for (int i = 0; i < trajectory.n_keyframes; i++) {
    trajectory.keyframes[i].p_fix.fill(true);
  }
  trajectory.keyframes[1].p_fix.fill(false);
  // trajectory.keyframes[0].v_fix.fill(true);
  // trajectory.keyframes[0].v_fix(0) = false;

  ArgIdxPV arg_pv;
  initialize_argPV(trajectory, arg_pv);
  int n_args = arg_pv.n_argP + arg_pv.n_argV;
  Vecxd arg(n_args * 3);
  Vecxd arg_solver(n_args * 3);
  select_optimize_argPV<3>(trajectory, arg_pv, arg);

  trajectory.keyframes[2].p_fix.fill(false);
  arg_pv.keyframe_idx[2].P_idx_in_arg = arg_pv.keyframe_idx[1].P_idx_in_arg;
  arg_pv.keyframe_fixed_idx[2].P_idx_in_arg =
      arg_pv.keyframe_fixed_idx[1].P_idx_in_arg;
  // for (int i = 1; i < n_verts; i++) {
  //   arg_pv.keyframe_idx[0].V_idx_in_arg(i) =
  //       arg_pv.keyframe_idx[0].V_idx_in_arg(0);
  // }
  // trajectory.keyframes[0].v_fix.fill(false);

  // =========================== optimize ===========================
  Vecxd f(n_verts * 3);
  Vecxd jacobian(n_verts * 3);
  Vecxd arg_J(n_args * 3);
  SparseMatd arg_H(n_args * 3, n_args * 3);
  SparseMatd sample_arg_J(n_verts * 3, n_args * 3);
  Eigen::SimplicialLDLT<SparseMatd> solver;

  // ============= regularization ===============
  Vec3d gravity(0.0, 0.0, 0.0);
  DataManager<3> data(vert_mass, damping_alpha, tm.tets, gravity);
  ArgSelection<3> arg_selection(trajectory, arg_pv);
  BezierMatrix<3> optimizer(trajectory, arg_pv, sample_batch, pd_solver,
                            arg_selection, data);

  SparseMatd sum_lhs_hessian_extended;
  SparseMatd S_ext;
  Vecxd arg_select_C_vec;
  {
    extendSparseMatrixByKronecker<3>(optimizer.sum_lhs_hessian,
                                     sum_lhs_hessian_extended);
    extendSparseMatrixByKronecker<3>(optimizer.arg_selection.S, S_ext);
    arg_select_C_vec = Vecxd::Map(optimizer.arg_selection.C.data(),
                                  optimizer.arg_selection.C.size());
  }

  KeyframeEditingInterface keyframe_interface(scene, gui, trajectory, tm,
                                              material);
  keyframe_interface.setup_deformation_params(tm.tets, tet_mass, vert_mass,
                                              ext_force, hydro_stiffness,
                                              devia_stiffness, 0.01);

  // ================ implicit-eulerized optimization ===============

  std::vector<int> violated_verts;

  auto optimize_trajectory_PV = [&](int max_steps = 10) {
    MatxXd rhs_mat = optimizer.rhs;
    Vecxd rhs = Vecxd::Map(rhs_mat.data(), rhs_mat.size());

    for (int i = 0; i < n_verts; i++) {
      if (trajectory.keyframes[1].pos(i, 1) < 0.0 &&
          std::find(violated_verts.begin(), violated_verts.end(), i) ==
              violated_verts.end()) {
        violated_verts.push_back(i);
      }
    }
    int n_violated = violated_verts.size();
    std::cout << "Violation count: " << n_violated << std::endl;
    int ndim = sum_lhs_hessian_extended.rows();
    SparseMatd sum_hessian_constrained = sum_lhs_hessian_extended;
    sum_hessian_constrained.conservativeResize(ndim + n_violated,
                                               ndim + n_violated);
    Vecxd rhs_constrained(ndim + n_violated);
    rhs_constrained.setZero();
    rhs.head(ndim) = rhs;

    for (int i = 0; i < n_violated; i++) {
      int idx = arg_pv.keyframe_idx[1].P_idx_in_arg(violated_verts[i]);
      sum_hessian_constrained.insert(ndim + i, idx * 3 + 1) = 1.0;
      sum_hessian_constrained.insert(idx * 3 + 1, ndim + i) = 1.0;
    }
    optimizer.solver.analyzePattern(sum_hessian_constrained);
    optimizer.solver.factorize(sum_hessian_constrained);
    Vecxd result = optimizer.solver.solve(rhs_constrained);
    arg = result.head(n_args * 3);
    retrive_from_argPV<3>(trajectory, arg_pv, arg);

    int j = 0;
    for (int i = 0; i < n_violated; ++i) {
      if (result(n_args * 3 + i) < 0) {
        violated_verts[j++] = violated_verts[i];
      }
    }
    violated_verts.resize(j);
    std::cout << "KKT condition count: " << j << std::endl;
  };

  auto optimize_trajectory_T = [&]() {};

  DiffuseMesh visual_mesh = create_diffuse_mesh(material);
  add_render_func(scene, get_render_func(visual_mesh));
  construct_visual_tets(vtm.visual_verts, tm.verts, tm.tets);
  set_mesh_data(visual_mesh, vtm.visual_verts.cast<float>(), vtm.visual_faces);
  aphys::Edges box_edges = aphys::create_box_edges();
  box_edges.width = 0.8f;
  box_edges.color = {0, 0, 0};
  aphys::set_box_edges_data(box_edges, box);
  aphys::add_render_func(scene, aphys::get_render_func(box_edges));

  float start_time = glfwGetTime();
  float tau = total_time;
  std::string output_path =
      SIMTF_RESOURCES_PATH + cnslash + "../output" + cnslash;
  char filename[128] = "";
  add_gui_func(gui, [&]() {
    bool pressed = ImGui::Button("Optimize");
    if (pressed) {
      optimize_trajectory_PV();
    }
    bool pressed2 = ImGui::Button("Optimize T");
    if (pressed2) {
      optimize_trajectory_T();
    }
    bool play = ImGui::Button("Play");
    if (play) {
      start_time = glfwGetTime();
      tau = 0.0;
    }
    ImGui::InputText("filename", filename, 128);

    bool export_usd = ImGui::Button("Save");
    if (export_usd) {
      TrajectoryExporter exporter(trajectory, tm.tets, 20);
      exporter.SaveAnimationToUSD(output_path + filename + ".usdc");
    }
  });

  glfwSwapInterval(1);

  while (!glfwWindowShouldClose(window)) {
    tau = (glfwGetTime() - start_time) / (1.0 / total_time);
    glfwPollEvents();
    handle_gui_input(gui);

    if (tau < total_time) {
      int idx =
          sample_damped_tau_on_trajectory(tau, trajectory, v_p, v_vel, v_acce);

      construct_visual_tets(vtm.visual_verts, v_p, tm.tets);
      set_mesh_data(visual_mesh, vtm.visual_verts.cast<float>(),
                    vtm.visual_faces);
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
