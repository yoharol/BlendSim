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

  Timer::getInstance()->start("precompute");

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
  double devia_stiffness = 300.0f;
  double hydro_stiffness = 300.0f;

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
  KeyframeEditingInterface keyframe_interface(scene, gui, trajectory, tm,
                                              material);
  keyframe_interface.setup_deformation_params(tm.tets, tet_mass, vert_mass,
                                              ext_force, hydro_stiffness,
                                              devia_stiffness, 0.01);

  // ================ implicit-eulerized optimization ===============
  double h = 1.0 / 60.0;
  std::vector<SparseMatd> eulerized_hessians;
  std::vector<SparseMatd> hessians;
  SparseMatd sum_hessian;
  precompute_damped_eulerized_hessian_batch<3>(
      sample_batch, arg_pv, trajectory.T_between, vert_mass, pd_solver.L_ext,
      damping_alpha, hessians, eulerized_hessians, sum_hessian, h);
  SparseMatd sum_hessian_constrained = sum_hessian;
  Timer::getInstance()->pause("precompute");
  print_tet_info(tm);

  double prev_loss = 0.0;

  std::vector<int> violated_verts;

  auto optimize_trajectory_PV = [&](int max_steps = 10) {
    bool converged = false;
    int step = 0;
    while (!converged && step++ < max_steps) {
      std::cout << "start optimize" << std::endl;
      Timer::getInstance()->start("optimize");

      double Wsp = summarize_damped_eularized_jacobian_PV_with_pre<3>(
          sample_batch, arg_pv, trajectory, vert_mass, arg_J, hessians,
          eulerized_hessians, pd_solver.energy_jacobian_func, damping_alpha);
      std::cout << "Wsp: " << Wsp << std::endl;
      std::cout << "complete sampling" << std::endl;
      std::cout << arg_J.squaredNorm() << std::endl;
      if ((Wsp - prev_loss) * (Wsp - prev_loss) > 1e-2) {
        for (int i = 0; i < n_verts; i++) {
          if (trajectory.keyframes[1].pos(i, 1) < 0.0 &&
              std::find(violated_verts.begin(), violated_verts.end(), i) ==
                  violated_verts.end()) {
            violated_verts.push_back(i);
          }
        }
        int n_violated = violated_verts.size();
        sum_hessian_constrained.resize(0, 0);
        sum_hessian_constrained = sum_hessian;
        sum_hessian_constrained.conservativeResize(n_args * 3 + n_violated,
                                                   n_args * 3 + n_violated);
        Vecxd rhs(n_args * 3 + n_violated);
        rhs.head(n_args * 3) = arg_J;

        for (int i = 0; i < n_violated; i++) {
          int idx = arg_pv.keyframe_idx[1].P_idx_in_arg(violated_verts[i]);
          sum_hessian_constrained.insert(n_args * 3 + i, idx * 3 + 1) = 1.0;
          sum_hessian_constrained.insert(idx * 3 + 1, n_args * 3 + i) = 1.0;
          rhs(n_args * 3 + i) =
              trajectory.keyframes[1].pos(violated_verts[i], 1);  // y-0>=0
        }
        solver.analyzePattern(sum_hessian_constrained);
        solver.factorize(sum_hessian_constrained);

        // double sum_y = 0;
        // for (int i = 0; i < n_verts; i++) {
        //   sum_y += trajectory.keyframes[1].pos(i, 1);
        // }
        // rhs(n_args * 3) = (sum_y - 0.3);
        Timer::getInstance()->start("solver");
        prev_loss = Wsp;
        Vecxd search_direction = -solver.solve(rhs);
        arg = arg + search_direction.head(n_args * 3);

        int j = 0;
        for (int i = 0; i < n_violated; ++i) {
          if (search_direction(n_args * 3 + i) < 0) {
            violated_verts[j++] = violated_verts[i];
          }
        }
        violated_verts.resize(j);
        std::cout << "KKT condition count: " << j << std::endl;

        /*size_t j = 0;
        for (size_t i = 0; i < n_violated; i) {
          if (b[i] >= 0) {
            a[j++] = a[i];
          }
        }
        a.resize(j);*/

        Timer::getInstance()->pause("solver");
        Timer::getInstance()->start("update");
        retrive_from_argPV<3>(trajectory, arg_pv, arg);
        Timer::getInstance()->pause("update");
        Timer::getInstance()->pause("optimize");
        Timer::getInstance()->print_all();
        print_tet_info(tm);
      } else {
        converged = true;
        break;
      }
      std::cout << "end optimizing\n";
      keyframe_interface.reset_keyframes(trajectory);
    }
  };

  auto optimize_trajectory_T = [&]() {
    Timer::getInstance()->start("T optimize");
    int n_intervals = trajectory.n_keyframes - 1;
    Vecxd sum_J_T;
    sum_J_T.setZero();
    Vecxd sum_H_T;
    sum_H_T.setZero();
    MatxXd LHS(n_intervals, n_intervals);
    LHS.setZero();
    Vecxd rhs(n_intervals);
    rhs.setZero();

    summarize_damped_derivative_T<3>(
        sample_batch, trajectory, vert_mass, pd_solver.energy_jacobian_func,
        pd_solver.L_ext, sum_J_T, sum_H_T, damping_alpha);
    std::cout << "sum_J_T: " << sum_J_T.transpose() << std::endl;
    std::cout << "sum_H_T: " << sum_H_T.transpose() << std::endl;
    /*LHS(0, 0) = sum_H_T(0);
    LHS(1, 1) = sum_H_T(2);
    LHS(2, 0) = 1.0;
    LHS(2, 1) = 1.0;
    LHS(0, 2) = 1.0;
    LHS(1, 2) = 1.0;
    rhs(0) = -sum_J_T(0);
    rhs(1) = -sum_J_T(2);
    rhs(2) = 0.0;
    std::cout << LHS << std::endl;
    std::cout << rhs.transpose() << std::endl;
    Vecxd result = LHS.colPivHouseholderQr().solve(rhs);
    trajectory.T_between[0] += result(0);
    trajectory.T_between[2] += result(1);*/
    Vecxd search_direct = sum_J_T;
    search_direct(0) /= sum_H_T(0);
    search_direct(2) /= sum_H_T(2);
    trajectory.T_between[0] += -sum_J_T(0) / sum_H_T(0);
    trajectory.T_between[2] += -sum_J_T(2) / sum_H_T(2);
    // T_damped_line_search<3>(sample_batch, trajectory, sum_J_T, search_direct,
    //                         vert_mass, pd_solver.energy_jacobian_func,
    //                         damping_alpha);

    std::cout << "Time intervals: ";
    for (int k = 0; k < n_intervals; k++) {
      std::cout << trajectory.T_between[k] << " ";
    }
    update_trajectory_tau(trajectory);
    total_time = sum_vector(trajectory.T_between);
    std::cout << std::endl;
    // trajectory.tau_stamp[n_intervals] = tau;
    Timer::getInstance()->pause("T optimize");
    Timer::getInstance()->print_all();
    precompute_eulerized_hessian_batch<3>(
        sample_batch, arg_pv, trajectory.T_between, vert_mass, pd_solver.L_ext,
        hessians, eulerized_hessians, sum_hessian, h);
    solver.analyzePattern(sum_hessian);
    solver.factorize(sum_hessian);
  };

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
      SaveFrameAnimatinoToUsd(trajectory, tm.tets,
                              output_path + filename + "_frame.usdc", 60);
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
