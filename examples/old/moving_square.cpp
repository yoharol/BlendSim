
#include "ArmorerPhys/RenderCore.h"
#include "ArmorerPhys/SimCore.h"
#include "igl/readOBJ.h"
#include "Eigen/SparseQR"
#include "Eigen/Sparse"

#include "simTF/spline.h"
#include "simTF/optimize.h"
#include "simTF/PD/pd.h"

int main() {
  using namespace aphys;
  const unsigned int SCR_WIDTH = 700;
  const unsigned int SCR_HEIGHT = 700;
  GLFWwindow* window =
      aphys::create_window(SCR_WIDTH, SCR_HEIGHT, "Moving Square");
  Scene scene = create_scene(default_light, default_camera);
  set_2d_camera(scene.camera, 0.0f, 1.0f, 0.0f, 1.0f);

  aphys::MatxXd verts;
  Matx3i face_indices;

  create_rectangle(0.45, 0.55, 1, 0.1, 0.2, 1, verts, face_indices);

  std::vector<MatxXd> key_frames(3);
  key_frames[0] = verts;
  key_frames[1] = verts;
  key_frames[1].col(1).array() += 0.5;
  key_frames[1].col(0).array() -= 0.3;
  key_frames[2] = key_frames[1];
  key_frames[2].col(0).array() += 0.6;
  key_frames[2].col(1).array() -= 0.4;

  std::vector<int> segments = {1, 1};
  std::vector<double> time_interval = {0.8, 0.8};
  double total_time = sum_vector(time_interval);

  SplineTrajectory trajectory;
  set_keyframe_bezier_interpolation(key_frames, segments, time_interval,
                                    trajectory);
  trajectory.keyframes[0].vel = (key_frames[1] - key_frames[0]) / 3.0;
  trajectory.keyframes[1].vel = (key_frames[1] - key_frames[0]) / 3.0;

  MatxXd v_p = trajectory.keyframes[1].pos;
  int n_verts = v_p.rows();
  int n_faces = face_indices.rows();
  MatxXd v_p_ref = v_p;
  MatxXd v_acce = v_p;
  Matx2i edge_indices;
  extract_edge(face_indices, edge_indices);
  Vecxd face_mass, vert_mass;
  compute_mesh_mass(v_p, face_indices, face_mass, vert_mass);
  DiagMatxXd v_M;
  set_diag_matrix(vert_mass, v_M, 2);

  /*

  aphys::DiffPDSolver2D pd_solver(v_p, v_p_ref, face_indices, face_mass,
                                  vert_mass, external_force, stiffness_hydro,
                                  stiffness_devia);*/

  // =========================== prepare sample ===========================
  SplineTrajectory traj_solver;
  copy_trajectory(trajectory, traj_solver);
  std::vector<int> n_samples_interval = {20};
  SampleBatch sample_batch;
  build_sample_batch(n_samples_interval, sample_batch, traj_solver);

  // =========================== set fixed incides ===========================
  Vecxb& k0p = trajectory.keyframes[0].p_fix;
  k0p.fill(true);
  Vecxb& k1p = trajectory.keyframes[1].p_fix;
  k1p.fill(true);
  Vecxb& k2p = trajectory.keyframes[2].p_fix;
  k2p.fill(true);

  // =========================== prepare argPV ===========================
  ArgIdxPV arg_pv;
  initialize_argPV(trajectory, arg_pv);
  int n_args = arg_pv.n_argP + arg_pv.n_argV;

  Vecxd arg(n_args * 2);
  Vecxd arg_solver(n_args * 2);
  select_optimize_argPV<2>(trajectory, arg_pv, arg);

  // =========================== Regularize ===========================
  Vec2d gravity(0.0, 0.0);
  aphys::MatxXd external_force;
  aphys::generate_gravity_force(gravity, vert_mass, external_force);
  double stiffness_hydro = 2.0;
  double stiffness_devia = 2.0;
  aphys::ProjectiveDynamicsSolver<2> pd_solver(
      v_p, v_p_ref, face_indices, face_mass, vert_mass, external_force,
      1.0 / 60.0, 1.0, 1.0);
  aphys::Regularizer<2> reg(n_verts);

  // for (int i = 0; i < trajectory.n_keyframes; i++) {
  //   trajectory.keyframes[i].vel.setOnes(n_verts, 2);
  // }

  SparseMatd sum_HR;
  reg.compute_regularize_gradient(trajectory, sum_HR);
  MatxXd rhs(n_verts * trajectory.n_keyframes * 2, 2);
  rhs.setZero();

  MatxXd new_arg;

  reg.regularize(trajectory, sum_HR, rhs, arg_pv, new_arg);

  apply_arg_in_solver(arg_pv, traj_solver, new_arg, trajectory);

  Points cp = create_points();
  set_points_data(cp, v_p.cast<float>(), MatxXf());
  cp.color = RGB(71, 147, 175);
  cp.point_size = 4.0f;
  Edges ce = create_edges();
  set_edges_data(ce, v_p.cast<float>(), edge_indices, MatxXf());
  ce.color = aphys::RGB(50, 44, 43);
  Lines cd = create_deriv_lines();
  cd.color = RGB(255, 0, 0);
  Lines cj = create_deriv_lines();
  cj.color = RGB(0, 255, 0);
  add_render_func(scene, get_render_func(cp));
  add_render_func(scene, get_render_func(ce));
  add_render_func(scene, get_render_func(cd));
  add_render_func(scene, get_render_func(cj));

  glfwSwapInterval(1);

  float start_time = glfwGetTime();
  float tau = 0.0;
  while (!glfwWindowShouldClose(window)) {
    tau = (glfwGetTime() - start_time);
    /*if (tau > 1.0) {
      if (true) {
        std::cout << "start optimize" << std::endl;
        for (int _ = 0; _ < 24; _++) {
          summarize_jacobian_PV<2>(
              sample_batch, arg_pv, trajectory, vert_mass, v_InvM, mass_tilde,
              dFdx, v_p, v_acce, f, jacobian, arg_J, arg_H, sample_arg_J,
              pd_solver.energy_jacobian_func, pd_solver.energy_hessian_func);
          if (arg_J.squaredNorm() > 1e-4) {
            std::cout << arg_J.squaredNorm() << std::endl;
            solver.compute(arg_H);
            Vecxd search_direction = -solver.solve(arg_J);
            line_search<2>(sample_batch, arg, search_direction, arg_solver,
                           arg_J, v_p, v_acce, jacobian, f, vert_mass, v_InvM,
                           mass_tilde, trajectory, arg_pv, traj_solver,
                           pd_solver.energy_jacobian_func);
            retrive_from_argPV<2>(trajectory, arg_pv, arg);
          }
        }
      }
      start_time = glfwGetTime();
      tau = 0.0;
      std::cout << "end optimizing" << std::endl;
    }*/

    /*for (SampleInfo& sample_info : sample_batch.samples) {
      if (tau >= sample_info.tau_stamp && tau <= sample_info.tau_next) {
        sample_dynamic_status(sample_info, trajectory, vert_mass, v_p, v_acce,
                              mass_tilde);
        pd_solver.energy_jacobian_func(v_p, jacobian);
        compute_force_residual<2>(n_verts, f, v_InvM, jacobian, mass_tilde,
                                  v_acce, vert_mass);
        MatxXd f_mat = Eigen::Map<MatxXd>(f.data(), n_verts, 2);
        set_deriv_lines_data(cd, v_p.cast<float>(), f_mat.cast<float>(), 2.0f,
                             MatxXf());
        MatxXd jacobian_mat = Eigen::Map<MatxXd>(jacobian.data(), n_verts, 2);
        set_deriv_lines_data(cj, v_p.cast<float>(), jacobian_mat.cast<float>(),
                             2.0f, MatxXf());
        break;
      }
    }*/

    if (tau > total_time) {
      start_time = glfwGetTime();
      tau = 0.0;
    }

    sample_tau_on_trajectory(tau, trajectory, v_p, v_acce);
    set_points_data(cp, v_p.cast<float>(), MatxXf());
    set_edges_data(ce, v_p.cast<float>(), edge_indices, MatxXf());

    /*pd_solver.energy_jacobian_func(v_p, jacobian);
    compute_force_residual<2>(n_verts, f, jacobian, mass_tilde, v_acce,
                              vert_mass);
    MatxXd f_mat = MatxXd::Map(f.data(), v_p.rows(), v_p.cols());

    set_deriv_lines_data(cd, v_p.cast<float>(), f_mat.cast<float>(), 1.0f,
                         MatxXf());*/

    aphys::set_background_RGB({244, 244, 244});
    render_scene(scene);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  glfwTerminate();
}
