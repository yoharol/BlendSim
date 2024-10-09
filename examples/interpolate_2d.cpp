#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "ArmorerPhys/RenderCore.h"
#include "ArmorerPhys/SimCore.h"
#include "ArmorerPhys/sim/fem.h"
#include "ArmorerPhys/sim/pd.h"

const unsigned int SCR_WIDTH = 700;
const unsigned int SCR_HEIGHT = 700;

int main() {
  GLFWwindow* window =
      aphys::create_window(SCR_WIDTH, SCR_HEIGHT, "2D interpolate");
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

  aphys::readOBJ(std::string(SIMTF_RESOURCES_PATH) + "/2d/ref.obj", v_p_ref,
                 face_indices, 2);
  aphys::readOBJ(std::string(SIMTF_RESOURCES_PATH) + "/2d/donut.obj", v_donut,
                 face_indices, 2);
  aphys::readOBJ(std::string(SIMTF_RESOURCES_PATH) + "/2d/twist1.obj", v_twist,
                 face_indices, 2);
  v_p = v_p_ref;

  aphys::extract_edge(face_indices, edge_indices);

  // ===================== prepare simulation ======================
  aphys::Vecxd face_mass, vert_mass;
  aphys::MatxXd external_force;
  aphys::compute_mesh_mass(v_p_ref, face_indices, face_mass, vert_mass, 1000.0);
  aphys::Vecxd gravity(2);
  gravity << 0.0, 0.0;
  aphys::generate_gravity_force(gravity, vert_mass, external_force);
  aphys::MatxXd source_B;
  aphys::NeoHookeanFEM2D::project_B(v_p, v_p_ref, face_indices, source_B);
  // aphys::ARAPTargetShape2D idle_shape(v_p, face_indices, source_B);
  aphys::ARAPTargetShape2D donut_shape(v_donut, face_indices, source_B);
  aphys::ARAPTargetShape2D twist_shape(v_twist, face_indices, source_B);

  for (int i = 100; i < face_indices.rows(); i++) {
    if (donut_shape.rotate_angle(i) < 0.0)
      donut_shape.rotate_angle(i) += 2.0 * M_PI;
  }
  aphys::MatxXd target_F = source_B;
  aphys::Vecxi fixed_verts(1);
  fixed_verts << 0;
  aphys::MatxXd fixed_pos(1, 2);
  fixed_pos.row(0) = v_p_ref.row(0);
  aphys::ARAPInterpolate2D arap_solver(v_p, v_p_ref, face_indices, face_mass,
                                       vert_mass, external_force, 1.0, 0.0,
                                       fixed_verts);

  auto arap_interpolate = [&](double w) {
    aphys::MatxXd interpolated_S =
        w * twist_shape.S + (1.0 - w) * donut_shape.S;
    aphys::Vecxd interpolate_rotate =
        w * twist_shape.rotate_angle + (1.0 - w) * donut_shape.rotate_angle;
    aphys::ARAPTargetShape2D::recover_deformation_map(
        interpolated_S, interpolate_rotate, target_F);
    arap_solver.solver_static_shape(v_p, face_indices, fixed_pos, target_F);
  };

  auto linear_interpolate = [&](double w) {
    v_p = w * v_twist + (1.0 - w) * v_donut;
  };

  // ===================== prepare render data =====================
  aphys::Points points = aphys::create_points();
  points.point_size = 3.0f;

  aphys::Edges edges = aphys::create_edges();
  edges.color = aphys::RGB(0, 0, 0);
  edges.width = 1.0f;

  aphys::set_points_data(points, v_p.cast<float>(), aphys::MatxXf());
  aphys::set_edges_data(edges, v_p.cast<float>(), edge_indices,
                        aphys::MatxXf());

  aphys::add_render_func(scene, aphys::get_render_func(points));
  aphys::add_render_func(scene, aphys::get_render_func(edges));

  bool checkboxes[3] = {true, false};
  int interpolation_mode = 0;

  aphys::add_gui_func(gui, [&]() {
    if (ImGui::Checkbox("Linear", &checkboxes[0])) {
      interpolation_mode = 0;
      checkboxes[0] = true;
      checkboxes[1] = false;
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("ARAP", &checkboxes[1])) {
      interpolation_mode = 1;
      checkboxes[0] = false;
      checkboxes[1] = true;
    }
  });

  double interpolate_weight = 0.0;
  float weightf = 0.0f;

  aphys::add_gui_func(gui, [&]() {
    bool new_weight =
        ImGui::SliderFloat("Interpolate weight", &weightf, 0.0f, 1.0f);
    interpolate_weight = weightf;
    if (new_weight) {
      if (interpolation_mode == 0)
        linear_interpolate(interpolate_weight);
      else if (interpolation_mode == 1)
        arap_interpolate(interpolate_weight);
      aphys::set_points_data(points, v_p.cast<float>(), aphys::MatxXf());
      aphys::set_edges_data(edges, v_p.cast<float>(), edge_indices,
                            aphys::MatxXf());
    }
  });

  // ===================== main loop =====================
  glfwSwapInterval(1);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    aphys::handle_gui_input(gui);
    aphys::set_background_RGB({244, 244, 244});
    aphys::render_scene(scene);
    aphys::render_gui(gui);

    glfwSwapBuffers(window);
  }
  glfwTerminate();
}
