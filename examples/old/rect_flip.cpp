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
  }
  int n_verts = tm.verts.rows();
  int n_faces = tm.faces.rows();
  int n_tets = tm.tets.rows();

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

  std::vector<MatxXd> key_frames(5);
  for (int i = 0; i < 5; i++) {
    key_frames[i] = tm.verts;

    Mat4d affine;
    affine.setIdentity();
    Vec3d add_vert(0.0, 0.0, 0.0);

    if (i == 1) {
      Mat3d rot = Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX())
                      .toRotationMatrix();
      affine.block(0, 0, 3, 3) = rot;
      add_vert << 0.0, 3.0, -1.0;
    }
    if (i == 2) {
      Mat3d rot =
          Eigen::AngleAxisd(-M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();
      affine.block(0, 0, 3, 3) = rot;
      add_vert << 0.0, 3.0, -4.0;
    }
    if (i == 3) {
      Mat3d rot = Eigen::AngleAxisd(-3.0 * M_PI / 2.0, Eigen::Vector3d::UnitX())
                      .toRotationMatrix();
      affine.block(0, 0, 3, 3) = rot;
      add_vert << 0.0, 3.0, -5.0;
    }
    if (i == 4) {
      Mat3d rot = Eigen::AngleAxisd(-2.0 * M_PI, Eigen::Vector3d::UnitX())
                      .toRotationMatrix();
      affine.block(0, 0, 3, 3) = rot;
      add_vert << 0.0, 0.0, -8.0;
    }

    for (int j = 0; j < tm.verts.rows(); j++) {
      Vec4d v;
      v << tm.verts.row(j).transpose(), 1.0;
      v = affine * v;
      key_frames[i].row(j) = v.head(3).transpose() + add_vert.transpose();
    }
  }

  std::vector<int> segments = {1, 1, 1, 1};
  std::vector<double> time_interval = {0.3, 0.3, 0.3, 0.3};
  std::vector<int> n_samples_interval = {10, 10, 10, 10};

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
  double devia_stiffness = 150.0f;
  double hydro_stiffness = 150.0f;
  double dt = 1.0 / 60.0;
  double damping_alpha = 1.0;

  ProjectiveDynamicsSolver<3> pd_solver(v_p, v_ref, tm.tets, tet_mass,
                                        vert_mass, ext_force, dt,
                                        hydro_stiffness, devia_stiffness);

  // =========================== set fixed incides ===========================

  Vecxi top_n_bottom(top_face_indices.size() + bottom_face_indices.size());
  top_n_bottom << top_face_indices, bottom_face_indices;
  for (int i = 0; i < bottom_face_indices.size(); i++) {
    int idx = bottom_face_indices(i);
    for (int j = 0; j < trajectory.n_keyframes; j++) {
      trajectory.keyframes[j].p_fix[idx] = 1;
      // trajectory.keyframes[j].v_fix[idx] = 1;
    }
  }
  for (int i = 0; i < bottom_face_indices.size(); i++) {
    int idx = bottom_face_indices(i);
    for (int j = 0; j < trajectory.n_keyframes; j++) {
      trajectory.keyframes[j].v_fix[idx] = 1;
    }
  }

  trajectory.keyframes[0].p_fix.fill(true);
  trajectory.keyframes[4].p_fix.fill(true);
  trajectory.keyframes[4].v_fix.fill(true);

  ArgIdxPV arg_pv;
  initialize_argPV(trajectory, arg_pv);
  int n_args = arg_pv.n_argP + arg_pv.n_argV;
  Vecxd arg(n_args * 3);
  Vecxd arg_solver(n_args * 3);

  trajectory.keyframes[4].v_fix = trajectory.keyframes[0].v_fix;
  arg_pv.keyframe_idx[4].V_idx_in_arg = arg_pv.keyframe_idx[0].V_idx_in_arg;
  arg_pv.keyframe_fixed_idx[4].V_idx_in_arg =
      arg_pv.keyframe_fixed_idx[0].V_idx_in_arg;

  select_optimize_argPV<3>(trajectory, arg_pv, arg);

  // =========================== optimize ===========================
  Vecxd f(n_verts * 3);
  Vecxd jacobian(n_verts * 3);
  Vecxd arg_J(n_args * 3);
  SparseMatd arg_H(n_args * 3, n_args * 3);
  SparseMatd sample_arg_J(n_verts * 3, n_args * 3);
  Eigen::SimplicialLDLT<SparseMatd> solver;

  // ================ implicit-eulerized optimization ===============
  double h = 1.0 / 30.0;
  std::vector<SparseMatd> eulerized_hessians;
  std::vector<SparseMatd> hessians;
  SparseMatd sum_hessian;
  precompute_damped_eulerized_hessian_batch<3>(
      sample_batch, arg_pv, trajectory.T_between, vert_mass, pd_solver.L_ext,
      damping_alpha, hessians, eulerized_hessians, sum_hessian, h);
  solver.analyzePattern(sum_hessian);
  solver.factorize(sum_hessian);

  double prev_loss = 0.0;

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
        Timer::getInstance()->start("solver");
        prev_loss = Wsp;
        Vecxd search_direction = -solver.solve(arg_J);
        arg = arg + search_direction;
        Timer::getInstance()->pause("solver");
        Timer::getInstance()->start("update");
        retrive_from_argPV<3>(trajectory, arg_pv, arg);
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

  double total_time = sum_vector(trajectory.T_between);
  float start_time = glfwGetTime();
  float tau = total_time;
  bool playing = false;

  auto optimize_trajectory_T = [&]() {
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
    Vecxd result = LHS.colPivHouseholderQr().solve(rhs);
    double tau = 0.0;
    Vecxd search_direc = result.head(n_intervals);
    std::cout << search_direc.transpose() << std::endl;

    // T_damped_line_search<3>(sample_batch, trajectory, sum_J_T, search_direc,
    //                         vert_mass, pd_solver.energy_jacobian_func,
    //                         damping_alpha);

    std::cout << "Time intervals: ";
    for (int k = 0; k < n_intervals; k++) {
      trajectory.T_between[k] += result(k);
      std::cout << trajectory.T_between[k] << " ";
    }
    update_trajectory_tau(trajectory);
    total_time = sum_vector(trajectory.T_between);
    std::cout << std::endl;
    precompute_eulerized_hessian_batch<3>(
        sample_batch, arg_pv, trajectory.T_between, vert_mass, pd_solver.L_ext,
        hessians, eulerized_hessians, sum_hessian, h);
    solver.analyzePattern(sum_hessian);
    solver.factorize(sum_hessian);
  };

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
      optimize_trajectory_T();
    }
    bool play = ImGui::Checkbox("Play", &playing);
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
      vtm.visual_verts.rowwise() -= vtm.visual_verts.colwise().mean();
      set_color_mesh_data(color_mesh, vtm.visual_verts.cast<float>(),
                          vtm.visual_faces, vtm.visual_colors);
    } else if (playing) {
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