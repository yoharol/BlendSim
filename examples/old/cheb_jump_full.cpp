
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
          {242, 242, 242},       // light color
          {109, 105, 188},       // ambient color
          {-6.0f, 0.35f, -2.0f}  // light position
      },
      aphys::create_camera(
          {8.44f, 3.13f, -4.57f},               // camera position
          {0.0f, 0.5f, 0.0f},                   // camera target
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

  std::string cnslash = "/";
  std::string assets_path = SIMTF_RESOURCES_PATH + cnslash + "cheb" + cnslash;

  LBSModel lbs_model(assets_path + "cheb_ref_points.dmat",  //
                     "",                                    //
                     assets_path + "cheb_bone_edges.dmat",  //
                     assets_path + "cheb_bbw_weight.dmat",  //
                     assets_path + "cheb_c_weight.dmat");

  {
    std::string ref_file = assets_path + "cheb.mesh";
    igl::readMESH(ref_file, tm.verts, tm.tets, tm.faces);
    v_ref = tm.verts;
    v_p = v_ref;
    v_acce = v_ref;
    v_acce.setZero();
    extract_visual_tets_surfaces(tm.tets, tm.verts, vtm.visual_faces);
    std::cout << "n_verts: " << tm.verts.rows() << std::endl;
    std::cout << "n_tets: " << tm.tets.rows() << std::endl;
  }

  lbs_model.load_anim_data(assets_path + "cheb_selected_anim.dmat");
  lbs_model.initialize_lbs_weights(tm.verts);

  int n_verts = tm.verts.rows();
  int n_faces = tm.faces.rows();
  int n_tets = tm.tets.rows();

  std::vector<MatxXd> key_frames(10);
  for (int i = 0; i < key_frames.size(); i++) {
    lbs_model.load_frame_data(i);

    MatxXd T = lbs_model.T;
    RowVec3d add_dir;
    add_dir << -0.05, 0.0, 0.08;
    if (i > 1)
      for (int j = 0; j < T.rows() / 4; j++)
        T.row(j * 4 + 3) += (i - 1) * add_dir;

    key_frames[i] = lbs_model.lbs_weights * T;
    if (i > 0) {
      if (i % 2 == 0) {
        key_frames[i].col(1).array() += 0.08;
      } else {
        for (int j = 0; j < T.rows() / 4; j++) {
          MatxXd affine = T.block(j * 4, 0, 4, 3).transpose();
          MatxXd full_affine(4, 4);
          full_affine.block(0, 0, 3, 4) = affine;
          MatxXd compress(4, 4);
          compress.setIdentity();
          // compress(1, 1) = 0.8;
          full_affine = compress * full_affine;
          affine = full_affine.block(0, 0, 3, 4);
          T.block(j * 4, 0, 4, 3) = affine.transpose();
        }
        key_frames[i] = lbs_model.lbs_weights * T;
      }
    }
    key_frames[i] *= 10.0;
  }

  tm.verts *= 10.0;
  v_p = tm.verts;
  v_ref = tm.verts;

  std::vector<int> segments = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<double> time_interval(9);
  for (double& t : time_interval) {
    t = 0.3;
  }

  double total_time = sum_vector(time_interval);
  std::vector<int> n_samples_interval = {10, 10, 10, 10, 10, 10, 10, 10, 10};

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
  double devia_stiffness = 270.0f;
  double hydro_stiffness = 270.0f;
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
    trajectory.keyframes[i].p_fix.fill(1);
  }
  // trajectory.keyframes[0].v_fix.fill(1);
  // trajectory.keyframes[key_frames.size() - 1].v_fix.fill(1);

  ArgIdxPV arg_pv;
  initialize_argPV(trajectory, arg_pv);
  int n_args = arg_pv.n_argP + arg_pv.n_argV;
  Vecxd arg(n_args * 3);
  Vecxd arg_solver(n_args * 3);
  select_optimize_argPV<3>(trajectory, arg_pv, arg);

  // =========================== optimize ===========================
  Vecxd f(n_verts * 3);
  Vecxd jacobian(n_verts * 3);
  Vecxd arg_J(n_args * 3);
  SparseMatd arg_H(n_args * 3, n_args * 3);
  SparseMatd sample_arg_J(n_verts * 3, n_args * 3);
  Eigen::SimplicialLDLT<SparseMatd> solver;

  // =========================== regularization
  /*Regularizer<3> reg(n_verts);
  SparseMatd sum_HR;
  reg.compute_regularize_gradient(trajectory, sum_HR);
  MatxXd rhs(n_verts * trajectory.n_keyframes * 2, 3);
  rhs.setZero();
  MatxXd new_arg;
  reg.regularize(trajectory, sum_HR, rhs, arg_pv, new_arg);
  // retrive_from_argPV<3>(trajectory, arg_pv, new_arg);
  apply_arg_in_solver(arg_pv, trajectory, new_arg, trajectory);*/

  // =========================== implicit-eulerized optimization
  double h = 1.0 / 60.0;
  std::vector<SparseMatd> eulerized_hessians;

  // =========================== pre-computation ===========================
  std::vector<SparseMatd> hessians;
  SparseMatd sum_hessian;
  precompute_damped_eulerized_hessian_batch<3>(
      sample_batch, arg_pv, trajectory.T_between, vert_mass, pd_solver.L_ext,
      damping_alpha, hessians, eulerized_hessians, sum_hessian, h);
  solver.analyzePattern(sum_hessian);
  solver.factorize(sum_hessian);

  std::cout << "testtest\n";

  KeyframeEditingInterface keyframe_interface(scene, gui, trajectory, tm,
                                              material);
  keyframe_interface.setup_deformation_params(tm.tets, tet_mass, vert_mass,
                                              ext_force, hydro_stiffness,
                                              devia_stiffness, 0.03);

  DiffuseMesh origin_mesh = create_diffuse_mesh(material);
  add_render_func(scene, get_render_func(origin_mesh));
  DiffuseMesh wire_mesh = create_diffuse_mesh(wire_material);
  add_render_func(scene, get_render_func(wire_mesh), true, true);

  double prev_loss = 0.0;

  auto optimize_trajectory_PV = [&](int max_steps = 100) {
    // reg.regularize(trajectory, sum_HR, rhs, arg_pv, new_arg);
    // apply_arg_in_solver(arg_pv, traj_solver, new_arg, trajectory);
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
        Timer::getInstance()->start("solver");
        prev_loss = Wsp;

        Vecxd search_direction = -solver.solve(arg_J);

        arg = arg + search_direction;
        Timer::getInstance()->pause("solver");

        Timer::getInstance()->start("update");

        retrive_from_argPV<3>(trajectory, arg_pv, arg);
        /*if (Wsp > prev_loss) {
          std::cout << "falied Wsp: " << Wsp << std::endl;
          converged = true;
          break;
        }*/
        Timer::getInstance()->pause("update");
        Timer::getInstance()->pause("optimize");

        Timer::getInstance()->print_all();
      } else {
        converged = true;
        break;
      }
      std::cout << "end optimizing\n";
    }
  };

  auto optimize_trajectory_T = [&]() {
    Timer::getInstance()->start("T optimize");
    int n_intervals = trajectory.n_keyframes - 1;
    Vecxd sum_J_T;
    sum_J_T.setZero();
    Vecxd sum_H_T;
    sum_H_T.setZero();
    MatxXd LHS(n_intervals + 1, n_intervals + 1);
    LHS.setZero();
    Vecxd rhs(n_intervals + 1);
    rhs.setZero();

    summarize_damped_derivative_T<3>(
        sample_batch, trajectory, vert_mass, pd_solver.energy_jacobian_func,
        pd_solver.L_ext, sum_J_T, sum_H_T, damping_alpha);
    std::cout << "sum_J_T: " << sum_J_T.transpose() << std::endl;
    std::cout << "sum_H_T: " << sum_H_T.transpose() << std::endl;
    for (int k = 0; k < n_intervals; k++) {
      LHS(k, k) = sum_H_T(k);
      rhs(k) = -sum_J_T(k);
      LHS(n_intervals, k) = 1.0;
      LHS(k, n_intervals) = 1.0;
    }
    LHS(n_intervals, n_intervals) = 0.0;
    rhs(n_intervals) = 0.0;
    Vecxd result = LHS.colPivHouseholderQr().solve(rhs);
    double tau = 0.0;
    Vecxd search_direc = result.head(n_intervals);

    T_damped_line_search<3>(sample_batch, trajectory, sum_J_T, search_direc,
                            vert_mass, pd_solver.energy_jacobian_func,
                            damping_alpha);

    std::cout << "Time intervals: ";
    for (int k = 0; k < n_intervals; k++) {
      // trajectory.tau_stamp[k] = tau;
      // trajectory.T_between[k] += result(k);
      // tau += trajectory.T_between[k];
      std::cout << trajectory.T_between[k] << " ";
    }
    update_trajectory_tau(trajectory);
    total_time = sum_vector(trajectory.T_between);
    std::cout << std::endl;

    Timer::getInstance()->pause("T optimize");
    Timer::getInstance()->print_all();
    // trajectory.tau_stamp[n_intervals] = tau;
    precompute_eulerized_hessian_batch<3>(
        sample_batch, arg_pv, trajectory.T_between, vert_mass, pd_solver.L_ext,
        hessians, eulerized_hessians, sum_hessian, h);
    solver.analyzePattern(sum_hessian);
    solver.factorize(sum_hessian);
  };

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
      optimize_trajectory_PV(5);
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
