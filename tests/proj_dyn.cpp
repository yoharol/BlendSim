#include <igl/readOBJ.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Eigen/Core>
#include <iostream>

#include "ArmorerPhys/RenderCore.h"
#include "ArmorerPhys/SimCore.h"
#include "ArmorerPhys/sim/pd.h"

// settings
const unsigned int SCR_WIDTH = 1000;
const unsigned int SCR_HEIGHT = 800;

int main() {
  GLFWwindow *window = aphys::create_window(
      SCR_WIDTH, SCR_HEIGHT, "Example13: 3D Projective Dynamics");

  aphys::DiffuseMaterial material{
      {90, 178, 255},   // diffuse color
      {202, 244, 255},  // specular color
      0.5f              // specular strength
  };
  aphys::Scene scene = aphys::create_scene(
      aphys::Light{
          {242, 242, 242},      // light color
          {109, 105, 188},      // ambient color
          {2.0f, 0.35f, -5.0f}  // light position
      },
      aphys::create_camera(
          {-5.0f, 0.0f, -8.0f},                 // camera position
          {0.0f, 0.5f, 0.0f},                   // camera target
          {0.0f, 1.0f, 0.0f},                   // camera up axis
          float(SCR_WIDTH) / float(SCR_HEIGHT)  // camera aspect
          ));
  aphys::Gui gui = aphys::create_gui(window, "gui");
  gui.width = 300;
  gui.height = 100;

  aphys::Vec3f diffuse_color(0.0f, 211.f / 255.f, 239.f / 255.f);
  aphys::add_gui_func(gui, [&diffuse_color]() {
    ImGui::Text("Diffuse Color:");
    ImGui::ColorEdit3("#c1", diffuse_color.data());
  });

  // ===================== create a rectangular prism =====================

  int n_verts;
  aphys::TetMesh tm;
  aphys::VisualTetMesh vtm;
  aphys::create_rectangular_prism(-0.5, 0.5, 4, 0.0, 2.0, 8, -0.5, 0.5, 4,
                                  tm.verts, tm.tets);
  n_verts = tm.verts.rows();
  aphys::extract_surface_from_tets(tm.verts.rows(), tm.tets, tm.faces);
  aphys::extract_visual_tets_surfaces(tm.tets, tm.verts, vtm.visual_faces);
  int n_faces = tm.faces.rows();

  aphys::Vecxi top_face_indices;
  aphys::get_axis_value_indices(2.0, 1, tm.verts, top_face_indices);
  aphys::Vecxi bottom_face_indices;
  aphys::get_axis_value_indices(0.0, 1, tm.verts, bottom_face_indices);

  aphys::MatxXf vert_color(n_verts, 3);
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

  // ===================== prepare simulation data =====================

  aphys::Vecxd vert_mass;
  aphys::Vecxd tet_mass;
  aphys::compute_tet_mass(tm.verts, tm.tets, tet_mass, vert_mass);

  int substep = 2;
  double dt = 1.0f / 60.0f / (double)substep;
  double devia_stiffness = 100.0f;
  double hydro_stiffness = 0.0f;
  double rho = 1000.0f;
  int dim = 3;
  aphys::Vecxd gravity(dim);
  gravity << 0.0f, -4.0f, 0.0f;
  aphys::MatxXd v_vel, v_pred, v_p_ref, v_cache, v_solver;
  v_p_ref = tm.verts;
  v_vel.resize(n_verts, dim);
  v_vel.setZero();
  v_pred.resize(n_verts, dim);
  v_solver.resize(n_verts, dim);
  aphys::Box3d box(-2.0f, 2.0f, -3.0f, 3.0f, -2.0f, 2.0f);
  aphys::MatxXd external_force;
  aphys::generate_gravity_force(gravity, vert_mass, external_force);

  // ================== prepare projective dynamics solver ==================
  aphys::ProjectiveDynamicsSolver3D pd_solver(
      tm.verts, v_p_ref, tm.tets, tet_mass, vert_mass, external_force, dt,
      hydro_stiffness, devia_stiffness);

  // ===================== prepare render =====================

  aphys::Edges box_edges = aphys::create_box_edges();
  box_edges.width = 0.8f;
  box_edges.color = {0, 0, 0};

  aphys::set_box_edges_data(box_edges, box);

  aphys::ColorMesh mesh = aphys::create_color_mesh(material);

  aphys::add_render_func(scene, aphys::get_render_func(mesh));
  aphys::add_render_func(scene, aphys::get_render_func(box_edges));

  aphys::set_wireframe_mode(false);

  while (!glfwWindowShouldClose(window)) {
    aphys::set_background_RGB(aphys::RGB(250, 240, 228));

    v_cache = tm.verts;
    aphys::ImplicitEuler::predict(v_pred, tm.verts, v_vel, external_force,
                                  vert_mass, dt);
    for (int i = 0; i < 20; i++) {
      pd_solver.localStep(tm.verts, tm.tets);
      pd_solver.globalStep(v_solver, v_pred);
      double error = (tm.verts - v_solver).norm();
      tm.verts = v_solver;
      if (error < 1e-4) {
        break;
      }
    }
    aphys::collision3d(box, tm.verts);
    aphys::ImplicitEuler::updateVelocity(v_vel, tm.verts, v_cache, dt);

    aphys::construct_visual_tets(vtm.visual_verts, tm.verts, tm.tets, 0.8);
    aphys::set_color_mesh_data(mesh, vtm.visual_verts.cast<float>(),
                               vtm.visual_faces, vtm.visual_colors);

    aphys::orbit_camera_control(window, scene.camera, 10.0, scene.delta_time);

    aphys::render_scene(scene);
    aphys::render_gui(gui);

    aphys::use_program(mesh.program);
    aphys::set_uniform_float3(mesh.program, "diffuseColor", diffuse_color);
    aphys::unuse_program();

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  aphys::delete_mesh(mesh);
  aphys::destroy_gui(gui);
  glfwTerminate();
  return 0;
}
