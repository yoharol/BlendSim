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

const unsigned int SCR_WIDTH = 700;
const unsigned int SCR_HEIGHT = 700;

int main() {
  using namespace aphys;

  GLFWwindow* window =
      aphys::create_window(SCR_WIDTH, SCR_HEIGHT, "2D blend sim");
  double bottom = -3.0f;
  double top = 3.0f;
  double left = -3.0f;
  double right = 3.0f;

  aphys::Scene scene =
      aphys::create_scene(aphys::default_light, aphys::default_camera);
  aphys::set_2d_camera(scene.camera, left, right, bottom, top);

  aphys::Gui gui = aphys::create_gui(window, "gui");
  gui.width = 300;
  gui.height = 50;

  // ================= create the outside edge of a capsule ==================
  aphys::MatxXd v_p, v_p_ref;
  aphys::MatxXd v_donut;
  aphys::MatxXd v_twist;
  aphys::Matx3i face_indices;
  aphys::Matx2i edge_indices;

  create_rectangle(-1.0, 1.0, 10, 0.0, 1.0, 10, v_p_ref, face_indices);
  Vecxi leftIdx;
  get_axis_value_indices(-1.0, 0, v_p_ref, leftIdx);
  Vecxi rightIdx;
  get_axis_value_indices(1.0, 0, v_p_ref, rightIdx);

  v_p = v_p_ref;
  int n_verts = v_p.rows();
  int n_faces = face_indices.rows();

  std::vector<MatxXd> key_frames(3);
  key_frames[0] = v_p_ref;
  key_frames[1] = v_p_ref;
  key_frames[2] = v_p_ref;
  for (int i = 0; i < n_verts; i++) {
    Mat2d rot = Eigen::Rotation2Dd(M_PI / 2.0).toRotationMatrix();
    key_frames[1].row(i) = (rot * key_frames[1].row(i).transpose()).transpose();
    key_frames[1].row(i) -= Vec2d(1.0, 0.0);
  }
  std::vector<int> segments = {1, 1};
  std::vector<double> time_interval = {0.5, 0.5};
  std::vector<int> n_samples_interval = {20, 20};

  SplineTrajectory trajectory;
  set_keyframe_bezier_interpolation(key_frames, segments, time_interval,
                                    trajectory);
  MatxXd v_acce = v_p;
  extract_edge(face_indices, edge_indices);
  Vecxd face_mass, vert_mass;
  compute_mesh_mass(v_p_ref, face_indices, face_mass, vert_mass);
  DiagMatxXd v_M;

  SampleBatch batch;
  build_sample_batch(n_samples_interval, batch, trajectory);

  for (int i = 0; i < leftIdx.size(); i++) {
    trajectory.keyframes[0].p_fix[leftIdx(i)] = true;
    trajectory.keyframes[1].p_fix[leftIdx(i)] = true;
    trajectory.keyframes.back().p_fix[leftIdx(i)] = true;
  }
  for (int i = 0; i < rightIdx.size(); i++) {
    trajectory.keyframes[0].p_fix[rightIdx(i)] = true;
    trajectory.keyframes[1].p_fix[rightIdx(i)] = true;
    trajectory.keyframes.back().p_fix[rightIdx(i)] = true;
  }

  ArgIdxPV arg_pv;
  initialize_argPV(trajectory, arg_pv);
  int n_args = arg_pv.n_argP + arg_pv.n_argV;

  Vecxd arg(n_args * 2);
  Vecxd arg_solver(n_args * 2);
  select_optimize_argPV<2>(trajectory, arg_pv, arg);

  Vec2d gravity(0.0, 0.0);
  aphys::MatxXd external_force;
  aphys::generate_gravity_force(gravity, vert_mass, external_force);
  double stiffness_hydro = 40.0;
  double stiffness_devia = 40.0;
  aphys::ProjectiveDynamicsSolver<2> pd_solver(
      v_p, v_p_ref, face_indices, face_mass, vert_mass, external_force,
      1.0 / 60.0, 1.0, 1.0);
  double damping_alpha = 1.0;

  Vecxd f(n_verts * 2);
  Vecxd jacobian(n_verts * 2);
  Vecxd arg_J(n_args * 2);
  SparseMatd arg_H(n_args * 2, n_args * 2);
  SparseMatd sample_arg_J(n_verts * 2, n_args * 2);
  Eigen::SimplicialLDLT<SparseMatd> solver;
  double h = 1.0 / 10.0;
  std::vector<SparseMatd> eulerized_hessians;
  std::vector<SparseMatd> hessians;
  SparseMatd sum_hessian;
  precompute_damped_eulerized_hessian_batch<2>(
      batch, arg_pv, trajectory.T_between, vert_mass, pd_solver.L_ext,
      damping_alpha, hessians, eulerized_hessians, sum_hessian, h);
  solver.analyzePattern(sum_hessian);
  solver.factorize(sum_hessian);

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

  DataManager<2> data(vert_mass, damping_alpha, face_indices, gravity);
  ArgSelection<2> arg_selection(trajectory, arg_pv);
  BezierMatrix<2> optimizer(trajectory, arg_pv, batch, pd_solver, arg_selection,
                            data);
  double prev_loss = 0.0;
  auto optimize_PV = [&](int max_steps = 10) {
    /*bool converged = false;
    int step = 0;
    while (!converged && step++ < max_steps) {
      double Wsp = summarize_damped_eularized_jacobian_PV_with_pre<2>(
          batch, arg_pv, trajectory, vert_mass, arg_J, hessians,
          eulerized_hessians, pd_solver.energy_jacobian_func, damping_alpha);
      std::cout << "Wsp: " << Wsp << std::endl;
      if ((Wsp - prev_loss) * (Wsp - prev_loss) > 1e-4) {
        prev_loss = Wsp;
        Vecxd search_direction = -solver.solve(arg_J);
        arg = arg + search_direction;
        retrive_from_argPV<2>(trajectory, arg_pv, arg);
      } else {
        converged = true;
        break;
      }
    }*/
    optimizer.local_step();
    optimizer.global_step();
  };

  add_gui_func(gui, [&]() {
    bool pressed = ImGui::Button("Optimize");
    if (pressed) {
      optimize_PV(1);
    }
  });

  glfwSwapInterval(1);

  float start_time = glfwGetTime();
  float total_time = sum_vector(trajectory.T_between);
  float tau = 0.0;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    handle_gui_input(gui);
    tau = (glfwGetTime() - start_time) / (2.0 / total_time);
    if (tau > total_time) {
      start_time = glfwGetTime();
      tau = 0.0;
    }

    sample_tau_on_trajectory(tau, trajectory, v_p, v_acce);
    set_points_data(cp, v_p.cast<float>(), MatxXf());
    set_edges_data(ce, v_p.cast<float>(), edge_indices, MatxXf());

    aphys::set_background_RGB({244, 244, 244});
    render_scene(scene);
    render_gui(gui);
    glfwSwapBuffers(window);
  }
  glfwTerminate();
}
