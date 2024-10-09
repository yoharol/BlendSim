#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Eigen/Core>
#include <iostream>

#include "igl/readOBJ.h"
#include "igl/writeMESH.h"
#include "ArmorerPhys/data/tetrahetralize.h"

#include "ArmorerPhys/RenderCore.h"
#include "ArmorerPhys/data/voxel_builder.h"
#include "ArmorerPhys/SimCore.h"
#include "ArmorerPhys/sim/pd.h"

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 800;

int main() {
  GLFWwindow *window =
      aphys::create_window(SCR_WIDTH, SCR_HEIGHT, "Tetegen and Barycentric");

  aphys::DiffuseMaterial material{
      {232, 141, 103},  // diffuse color
      {255, 255, 255},  // specular color
      0.5f              // specular strength
  };
  aphys::DiffuseMaterial tet_material{
      {0, 200, 200},  // diffuse color
      {0, 0, 0},      // specular color
      0.0f            // specular strength
  };
  aphys::Scene scene = aphys::create_scene(
      aphys::Light{
          {242, 242, 242},      // light color
          {100, 100, 100},      // ambient color
          {5.0f, 0.35f, -5.0f}  // light position
      },
      aphys::create_camera(
          {6.0f, 0.0f, -6.0f},                  // camera position
          {0.0f, 0.35f, 0.0f},                  // camera target
          {0.0f, 1.0f, 0.0f},                   // camera up axis
          float(SCR_WIDTH) / float(SCR_HEIGHT)  // camera aspect
          ));

  aphys::Vec3f diffuse_color(0.0f, 211.f / 255.f, 239.f / 255.f);

  aphys::MatxXd V0, V1;
  aphys::MatxXi F0, F1;

  std::string cnslash = "/";
  std::string ref_shape = "sphere/sphere.obj";
  std::string bind_shape, output_dir;
  bind_shape = "sphere/sphere_empty.obj";
  output_dir = "sphere/sphere.mesh";
  bool bind = false;

  igl::readOBJ(std::string(SIMTF_RESOURCES_PATH) + cnslash + ref_shape, V0, F0);
  if (bind)
    igl::readOBJ(std::string(SIMTF_RESOURCES_PATH) + cnslash + bind_shape, V1,
                 F1);

  aphys::TetMesh tm;
  aphys::tetgen(V0, F0, tm);

  igl::writeMESH(std::string(SIMTF_RESOURCES_PATH) + cnslash + output_dir,
                 tm.verts, tm.tets, tm.faces);

  aphys::VisualTetMesh vtm;
  aphys::extract_visual_tets_surfaces(tm.tets, tm.verts, vtm.visual_faces);

  aphys::MatxXd bc_weights;
  aphys::Vecxi bc_index;
  aphys::SparseMatd bind_mat;
  if (bind) {
    aphys::bind_to_tet(V1, tm, bc_index, bc_weights);
    aphys::generate_bind_mat(tm.verts.rows(), tm.tets, bc_index, bc_weights,
                             bind_mat);
  }

  // ===================== Simulation =====================

  int n_verts = tm.verts.rows();
  aphys::Vecxd vert_mass;
  aphys::Vecxd tet_mass;
  aphys::compute_tet_mass(tm.verts, tm.tets, tet_mass, vert_mass);
  int substep = 2;
  double dt = 1.0f / 60.0f / (double)substep;
  double devia_stiffness = 50.0f;
  double hydro_stiffness = 30.0f;
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
  aphys::MatxXd external_force;
  aphys::generate_gravity_force(gravity, vert_mass, external_force);
  aphys::ProjectiveDynamicsSolver3D pd_solver(
      tm.verts, v_p_ref, tm.tets, tet_mass, vert_mass, external_force, dt,
      hydro_stiffness, devia_stiffness);

  aphys::Box3d box(-2.0f, 2.0f, -2.0f, 2.0f, -2.0f, 2.0f);

  aphys::DiffuseMesh mesh = aphys::create_diffuse_mesh(tet_material);
  aphys::DiffuseMesh fine_mesh = aphys::create_diffuse_mesh(material);
  aphys::DiffuseMesh tet_mesh = aphys::create_diffuse_mesh(tet_material);

  aphys::Edges box_edges = aphys::create_box_edges();
  box_edges.width = 0.8f;
  box_edges.color = {0, 0, 0};
  aphys::set_box_edges_data(box_edges, box);

  aphys::add_render_func(scene, aphys::get_render_func(mesh), true, false);
  aphys::add_render_func(scene, aphys::get_render_func(fine_mesh), true);
  aphys::add_render_func(scene, aphys::get_render_func(box_edges));

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
    if (bind) V1 = bind_mat * tm.verts;

    aphys::construct_visual_tets(vtm.visual_verts, tm.verts, tm.tets, 0.8);
    aphys::set_mesh_data(mesh, vtm.visual_verts.cast<float>(),
                         vtm.visual_faces);
    if (bind) aphys::set_mesh_data(fine_mesh, V1.cast<float>(), F0);

    aphys::orbit_camera_control(window, scene.camera, 10.0, scene.delta_time);

    aphys::render_scene(scene);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  aphys::delete_mesh(mesh);
  aphys::delete_mesh(fine_mesh);
  glfwTerminate();

  return 0;
}
