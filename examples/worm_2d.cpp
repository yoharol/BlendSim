#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Eigen/Core>
#include <iostream>

#include "ArmorerPhys/RenderCore.h"
#include "ArmorerPhys/SimCore.h"
#include "ArmorerPhys/sim/pd.h"
#include "igl/readTGF.h"
#include "igl/readDMAT.h"

#include "simTF/spline.h"
#include "simTF/PD/pd.h"
#include "simTF/optimize.h"
#include "simTF/renderer/renderer.h"
#include "simTF/lbs/model.h"

const unsigned int SCR_WIDTH = 700;
const unsigned int SCR_HEIGHT = 700;

int main() {
  GLFWwindow* window = aphys::create_window(
      SCR_WIDTH, SCR_HEIGHT, "Example7: interactive projective dynamics");
  double bottom = -0.5f;
  double top = 1.5f;
  double left = -0.5f;
  double right = 1.5f;

  aphys::Scene scene =
      aphys::create_scene(aphys::default_light, aphys::default_camera);
  aphys::set_2d_camera(scene.camera, left, right, bottom, top);
  aphys::Gui gui = aphys::create_gui(window, "gui");
  gui.width = SCR_WIDTH;
  gui.height = 80;

  // ===================== create a rectangle =====================
  aphys::MatxXd v_p;
  aphys::Matx3i face_indices;
  aphys::MatxXd lbs_weights;
  std::string cnslash = "/";
  std::string assets_path =
      SIMTF_RESOURCES_PATH + cnslash + "2d" + cnslash + "worm" + cnslash;
  igl::readDMAT(assets_path + "Weights.dmat", lbs_weights);
  igl::readOBJ(assets_path + "Mesh.obj", v_p, face_indices);
  v_p = v_p.leftCols(2);

  aphys::Vec3d refxy;
  refxy << 20.286886070344, 133.12241300448, 598.9168719408173;
  aphys::MatxXd con_v(5, 2);
  con_v << 410, 328,  //
      373, 183,       //
      258, 277,       //
      183, 176,       //
      69, 205;
  for (int i = 0; i < con_v.rows(); i++) {
    con_v(i, 0) = (con_v(i, 0) - refxy(0)) / refxy(2);
    con_v(i, 1) = (con_v(i, 1) - refxy(1)) / refxy(2);
  }
  aphys::Vecxi P(5);
  P << 0, 1, 2, 3, 4;
  aphys::Matx2i BE;

  aphys::Matx2i edge_indices;
  aphys::extract_edge(face_indices, edge_indices);
  std::cout << lbs_weights.topRows(3) << std::endl;

  aphys::LBSModel2D lbs_model(v_p, con_v, P, BE, lbs_weights);

  // ===================== prepare simulation data =====================
  int n_verts = v_p.rows();
  int substep = 1;
  double dt = 1.0f / 100.0f;
  double devia_stiffness = 10.0f;
  double hydro_stiffness = 10.0f;
  double rho = 1000.0f;
  int dim = 2;
  aphys::Vecxd gravity(dim);
  gravity << 0.0f, 0.0f;
  aphys::MatxXd v_vel, v_pred, v_p_ref, v_cache, v_solver;
  v_p_ref = v_p;
  v_vel.setZero();
  v_pred.resize(v_p.rows(), v_p.cols());
  v_solver.resize(v_p.rows(), v_p.cols());
  aphys::Vecxd vert_mass, face_mass;
  aphys::compute_mesh_mass(v_p_ref, face_indices, face_mass, vert_mass, rho);
  aphys::DiagMatxXd M(n_verts * 2);
  aphys::set_diag_matrix(vert_mass, M, 2);
  aphys::Box2d box(0.0f, 1.0f, 0.0f, 1.0f);
  aphys::MatxXd external_force;
  aphys::generate_gravity_force(gravity, vert_mass, external_force);

  // ================ prepare projective dynamics solver =====================
  aphys::ProjectiveDynamicsSolver2D pd_solver(
      v_p, v_p_ref, face_indices, face_mass, vert_mass, external_force, dt,
      hydro_stiffness, devia_stiffness);

  // ===================== prepare animation =====================
  int n_frames = 3;
  aphys::SplineTrajectory lbs_traj;
  double total_time;
  double damping_alpha = 1.0;
  aphys::SampleBatch sample_batch;
  {
    std::vector<aphys::MatxXd> key_frames(n_frames);
    aphys::MatxXd T_origin = lbs_model.T;
    lbs_model.set_transform(1, aphys::Vec2d(0.08, 0.02), 0.0);
    lbs_model.set_transform(4, aphys::Vec2d(0.0, 0.0), 0.3);
    key_frames[0] = lbs_model.T;
    key_frames[2] = lbs_model.T;
    lbs_model.T = T_origin;
    lbs_model.set_transform(1, aphys::Vec2d(-0.08, -0.02), 0.0);
    lbs_model.set_transform(4, aphys::Vec2d(0.0, 0.0), -0.7);
    key_frames[1] = lbs_model.T;
    lbs_model.T = T_origin;
    std::vector<int> segments(n_frames - 1);
    for (int i = 0; i < n_frames - 1; i++) {
      segments[i] = 1;
    }
    std::vector<double> time_interval(n_frames - 1);
    for (double& t : time_interval) {
      t = 0.09;
    }
    total_time = aphys::sum_vector(time_interval);
    aphys::set_keyframe_bezier_interpolation(key_frames, segments,
                                             time_interval, lbs_traj);
    std::vector<int> n_samples_interval(n_frames - 1);
    for (int& n : n_samples_interval) {
      n = 10;
    }
    aphys::build_sample_batch(n_samples_interval, sample_batch, lbs_traj);
  }
  int n_controls = lbs_traj.n_verts;
  aphys::LBSDataManager2D ldm(n_controls, pd_solver, lbs_model, v_p,
                              face_indices, M);
  double w0 = compute_lbs2d_wsp(sample_batch, n_verts, lbs_traj, pd_solver, ldm,
                                damping_alpha);
  std::cout << w0 << std::endl;

  for (int i = 0; i < n_frames; i++) {
    lbs_traj.keyframes[i].p_fix(5) = true;
    lbs_traj.keyframes[i].p_fix(14) = true;
    lbs_traj.keyframes[i].p_fix(13) = true;
  }
  lbs_traj.keyframes[2].p_fix.fill(true);
  lbs_traj.keyframes[2].v_fix.fill(true);

  aphys::ArgIdxPV arg_pv;
  initialize_argPV(lbs_traj, arg_pv);
  int n_args = arg_pv.n_argP + arg_pv.n_argV;
  aphys::Vecxd arg(n_args * 2);
  aphys::Vecxd arg_solver(n_args * 2);
  aphys::Vecxd arg_J(n_args * 2);
  aphys::select_optimize_argPV<2>(lbs_traj, arg_pv, arg);

  lbs_traj.keyframes[2].p_fix = lbs_traj.keyframes[0].p_fix;
  lbs_traj.keyframes[2].v_fix = lbs_traj.keyframes[0].v_fix;
  arg_pv.keyframe_idx[2].P_idx_in_arg = arg_pv.keyframe_idx[0].P_idx_in_arg;
  arg_pv.keyframe_fixed_idx[2].P_idx_in_arg =
      arg_pv.keyframe_fixed_idx[0].P_idx_in_arg;
  arg_pv.keyframe_idx[2].V_idx_in_arg = arg_pv.keyframe_idx[0].V_idx_in_arg;
  arg_pv.keyframe_fixed_idx[2].V_idx_in_arg =
      arg_pv.keyframe_fixed_idx[0].V_idx_in_arg;

  aphys::DataManager<2> data(vert_mass, damping_alpha, face_indices, gravity);
  aphys::ArgSelection<2> arg_selection(lbs_traj, arg_pv);
  aphys::BezierLBS2D optimizer(lbs_traj, arg_pv, sample_batch, pd_solver,
                               arg_selection, data, lbs_model);
  auto optimize_trajectory_PV = [&]() {
    optimizer.local_step();
    optimizer.global_step();
    std::cout << lbs_traj.keyframes[0].pos.row(5) << std::endl;
    double w0 = compute_lbs2d_wsp(sample_batch, n_verts, lbs_traj, pd_solver,
                                  ldm, damping_alpha);
    std::cout << w0 << std::endl;
  };
  add_gui_func(gui, [&]() {
    if (ImGui::Button("Optimize")) {
      optimize_trajectory_PV();
    }
  });

  // ===================== prepare render =====================
  aphys::Points control_point = aphys::create_points();
  aphys::set_points_data(control_point, con_v.cast<float>(), aphys::MatxXf());
  control_point.color = aphys::RGB(255, 0, 0);
  control_point.point_size = 3.0f;
  aphys::Edges edges = aphys::create_edges();
  aphys::set_edges_data(edges, v_p.cast<float>(), edge_indices,
                        aphys::MatxXf());
  edges.color = aphys::RGB(0, 0, 0);
  edges.width = 1.0f;
  aphys::add_render_func(scene, aphys::get_render_func(control_point));
  aphys::add_render_func(scene, aphys::get_render_func(edges));

  // ===================== prepare gui function =====================

  glfwSwapInterval(1);
  int frame = 0;
  double start_time = glfwGetTime();
  while (!glfwWindowShouldClose(window)) {
    // projective dynamcis solver
    // within the implicit euler framework
    /*for (int _ = 0; _ < substep; _++) {
      v_cache = v_p;
      aphys::ImplicitEuler::predict(v_pred, v_p, v_vel, external_force,
                                    vert_mass, dt);
      for (int i = 0; i < 100; i++) {
        pd_solver.localStep(v_p, face_indices);
        pd_solver.globalStep(v_solver, v_pred);
        double error = (v_solver - v_p).norm();
        v_p = v_solver;
        if (error < 1e-5) break;
      }
      aphys::collision2d(box, v_p);

      aphys::ImplicitEuler::updateVelocity(v_vel, v_p, v_cache, dt);
    }*/
    double tau = (glfwGetTime() - start_time) / (1.0 / total_time);
    if (tau > total_time) {
      tau = 0.0;
      start_time = glfwGetTime();
    }
    aphys::MatxXd sampled_T;
    aphys::MatxXd sampled_T_acce;
    aphys::MatxXd verts;
    sample_tau_on_trajectory(tau, lbs_traj, sampled_T, sampled_T_acce);
    verts = ldm.U * sampled_T;

    aphys::MatxXd new_con_v = con_v;
    for (int i = 0; i < con_v.rows(); i++) {
      aphys::RowVec3d cp;
      cp << con_v.row(i), 1.0;
      new_con_v.row(i) = cp * sampled_T.block(i * 3, 0, 3, 2);
    }

    aphys::set_edges_data(edges, verts.cast<float>(), edge_indices,
                          aphys::MatxXf());
    aphys::set_points_data(control_point, new_con_v.cast<float>(),
                           aphys::MatxXf());

    glfwPollEvents();
    aphys::handle_gui_input(gui);

    aphys::set_background_RGB({244, 244, 244});

    aphys::render_scene(scene);
    aphys::render_gui(gui);

    glfwSwapBuffers(window);
  }
  aphys::destroy_gui(gui);
  glfwTerminate();
}