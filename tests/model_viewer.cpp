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
#include "igl/writeDMAT.h"

#include "simTF/spline.h"
#include "simTF/PD/pd.h"
#include "simTF/optimize.h"
#include "simTF/optimize/damping_version.h"
#include "simTF/renderer/renderer.h"

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
          {6.44f, 0.33f, -1.57f},               // camera position
          {0.0f, 0.5f, 0.0f},                   // camera target
          {0.0f, 1.0f, 0.0f},                   // camera up axis
          float(SCR_WIDTH) / float(SCR_HEIGHT)  // camera aspect
          ));
  aphys::Vec3f diffuse_color(0.0f, 211.f / 255.f, 239.f / 255.f);
  aphys::Gui gui = aphys::create_gui(window, "gui");
  gui.width = 300;
  gui.height = 200;

  // =========================== prepare data ===========================
  MatxXd v_ref;
  MatxXd v_p, v_acce, v_vel;
  TetMesh tm;
  VisualTetMesh vtm;

  std::string cnslash = "/";
  std::string assets_path = SIMTF_RESOURCES_PATH + cnslash + "dino" + cnslash;
  {
    std::string ref_file = assets_path + "dino.mesh";
    igl::readMESH(ref_file, tm.verts, tm.tets, tm.faces);
    v_ref = tm.verts;
    v_p = v_ref;
    v_acce = v_ref;
    v_acce.setZero();
    extract_visual_tets_surfaces(tm.tets, tm.verts, vtm.visual_faces);
  }
  MatxXf vert_color(tm.verts.rows(), 3);
  Vecxi is_tail;
  igl::readDMAT(assets_path + "dino_is_tail.dmat", is_tail);
  for (int i = 0; i < tm.verts.rows(); i++) {
    if (is_tail(i)) {
      vert_color.row(i) << 1.0f, 0.0f, 0.0f;
    } else {
      vert_color.row(i) << 0.0f, 211.f / 255.f, 239.f / 255.f;
    }
  }

  construct_visual_tets(vtm.visual_verts, tm.verts, tm.tets);
  construct_visual_tets_color(vtm.visual_colors, vert_color, tm.tets);
  int n_verts = tm.verts.rows();
  int n_faces = tm.faces.rows();
  int n_tets = tm.tets.rows();

  std::vector<MatxXd> key_frames(10);
  for (int i = 0; i < key_frames.size(); i++) {
    std::string filename =
        assets_path + "dino_tet_" + std::to_string(i) + ".mesh";
    Matx3i F;
    Matx4i T;
    igl::readMESH(filename, key_frames[i], T, F);
  }

  MatxXd V0;
  MatxXi F0;
  igl::readOBJ(assets_path + "dino.obj", V0, F0);
  aphys::SparseMatd bind_mat;
  aphys::MatxXd bc_weights;
  aphys::Vecxi bc_index;
  igl::readDMAT(assets_path + "dino_weight_weight.dmat", bc_weights);
  igl::readDMAT(assets_path + "dino_weight_idx.dmat", bc_index);
  aphys::generate_bind_mat(tm.verts.rows(), tm.tets, bc_index, bc_weights,
                           bind_mat);

  ColorMesh mesh = create_color_mesh(material);
  add_render_func(scene, get_render_func(mesh));

  set_color_mesh_data(mesh, vtm.visual_verts.cast<float>(), vtm.visual_faces,
                      vtm.visual_colors);

  glfwSwapInterval(1);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    handle_gui_input(gui);

    aphys::set_background_RGB({244, 244, 244});
    aphys::orbit_camera_control(window, scene.camera, 10.0, scene.delta_time);
    render_scene(scene);
    render_gui(gui);
    glfwSwapBuffers(window);
  }
  destroy_gui(gui);
  glfwTerminate();
}