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
#include "igl/lbs_matrix.h"

#include "simTF/spline.h"
#include "simTF/PD/pd.h"
#include "simTF/optimize.h"
#include "simTF/optimize/damping_version.h"
#include "simTF/renderer/renderer.h"

void update_bone_rig(aphys::MatxXd& bc, const aphys::MatxXd& bc_ref,
                     const aphys::Vecxd& anim_frame_data,
                     const aphys::Matx2i& ce) {
  int n_cedges = ce.rows();
  aphys::MatxXd affines =
      aphys::MatxXd::Map(anim_frame_data.data(), n_cedges * 3, 4);
  for (int i = 0; i < n_cedges; i++) {
    aphys::MatxXd affine = affines.block(i * 3, 0, 3, 4);

    int id1 = ce(i, 0);
    int id2 = ce(i, 1);

    aphys::Vec4d bc1;
    bc1 << bc_ref.row(id1).transpose(), 1.0;
    aphys::Vec4d bc2;
    bc2 << bc_ref.row(id2).transpose(), 1.0;

    bc.row(id1) = (affine * bc1).transpose();
    bc.row(id2) = (affine * bc2).transpose();
  }
}

int main() {
  using namespace aphys;
  // =========================== prepare gl enviromnent
  // ===========================
  const unsigned int SCR_WIDTH = 800;
  const unsigned int SCR_HEIGHT = 800;
  GLFWwindow* window =
      aphys::create_window(SCR_WIDTH, SCR_HEIGHT, "Cheb Viewer");

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
  MatxXd c_p_ref;
  MatxXd c_p;
  Matx2i c_e;
  MatxXd anim_data;

  TetMesh tm;
  VisualTetMesh vtm;
  Matx3f tm_color;

  MatxXd bbw_weights;
  MatxXd cW;
  Eigen::MatrixXd W;

  std::string cnslash = "/";
  std::string assets_path = SIMTF_RESOURCES_PATH + cnslash + "cheb" + cnslash;

  Box3d box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

  igl::readDMAT(assets_path + "cheb_ref_points.dmat", c_p_ref);
  igl::readDMAT(assets_path + "cheb_bone_edges.dmat", c_e);
  igl::readDMAT(assets_path + "cheb_affine_anim.dmat", anim_data);
  igl::readMESH(assets_path + "cheb.mesh", tm.verts, tm.tets, tm.faces);
  igl::readDMAT(assets_path + "cheb_bbw_weight.dmat", bbw_weights);
  igl::readDMAT(assets_path + "cheb_c_weight.dmat", cW);
  igl::lbs_matrix(tm.verts, bbw_weights, W);

  MatxXd added_cp_ref;
  Matx2i added_ce;
  MatxXd added_bbw_weights;
  MatxXd added_cW;
  Eigen::MatrixXd added_W;

  igl::readDMAT(assets_path + "cheb_added_ref_points.dmat", added_cp_ref);
  igl::readDMAT(assets_path + "cheb_added_bone_edges.dmat", added_ce);
  igl::readDMAT(assets_path + "cheb_added_bbw_weight.dmat", added_bbw_weights);
  igl::readDMAT(assets_path + "cheb_added_c_weight.dmat", added_cW);
  igl::lbs_matrix(tm.verts, added_bbw_weights, added_W);

  extract_visual_tets_surfaces(tm.tets, tm.verts, vtm.visual_faces);
  tm_color.resize(tm.verts.rows(), 3);
  tm_color.setConstant(0.5f);

  c_p = c_p_ref;

  Points cheb_points = create_points();
  cheb_points.point_size = 5.0f;
  Edges cheb_edges = create_edges();
  aphys::Edges box_edges = aphys::create_box_edges();
  Points addeed_cheb_points = create_points();
  addeed_cheb_points.point_size = 5.0f;
  addeed_cheb_points.color = {0, 255, 255};
  Edges added_cheb_edges = create_edges();
  box_edges.width = 0.8f;
  box_edges.color = {0, 0, 0};
  aphys::set_box_edges_data(box_edges, box);
  aphys::add_render_func(scene, aphys::get_render_func(box_edges));

  construct_visual_tets(vtm.visual_verts, tm.verts, tm.tets);
  ColorMesh visual_mesh = create_color_mesh(material);
  add_render_func(scene, get_render_func(visual_mesh), true, true);
  construct_visual_tets(vtm.visual_verts, tm.verts, tm.tets);
  construct_visual_tets_color(vtm.visual_colors, tm_color, tm.tets);
  set_color_mesh_data(visual_mesh, vtm.visual_verts.cast<float>(),
                      vtm.visual_faces, vtm.visual_colors);

  add_render_func(scene, get_render_func(cheb_points), false);
  add_render_func(scene, get_render_func(cheb_edges), false);
  add_render_func(scene, get_render_func(addeed_cheb_points), false);
  add_render_func(scene, get_render_func(added_cheb_edges), false);

  glfwSwapInterval(1);

  int frame = 0;

  // reorgnize the animation affine matrix
  {
    int n_total_frames = anim_data.rows();
    aphys::MatxXd affine_anim_data(n_total_frames, c_e.rows() * 3 * 4);
    aphys::MatxXd added_affine_anim_data(n_total_frames,
                                         (c_e.rows() + 2) * 3 * 4);
    for (frame = 0; frame < n_total_frames; frame++) {
      aphys::Vecxd anim_frame_data = anim_data.row(frame);
      aphys::MatxXd affines =
          aphys::MatxXd::Map(anim_frame_data.data(), c_e.rows() * 3, 4);
      aphys::MatxXd reorg_affines(cW.rows() * 4, 3);
      aphys::MatxXd added_reorg_affines(added_cW.rows() * 4, 3);
      for (int i = 0; i < c_e.rows(); i++) {
        reorg_affines.block(i * 4, 0, 4, 3) =
            affines.block(i * 3, 0, 3, 4).transpose();
        added_reorg_affines.block(i * 4, 0, 4, 3) =
            affines.block(i * 3, 0, 3, 4).transpose();
      }
      added_reorg_affines.block((c_e.rows()) * 4, 0, 4, 3) =
          affines.block(2 * 3, 0, 3, 4).transpose();
      added_reorg_affines.block((c_e.rows() + 1) * 4, 0, 4, 3) =
          affines.block(2 * 3, 0, 3, 4).transpose();

      if (frame == 0) {
        std::cout << affines << std::endl << std::endl;
        std::cout << reorg_affines << std::endl;
      }

      affine_anim_data.row(frame) =
          Vecxd::Map(reorg_affines.data(), reorg_affines.size()).transpose();
      added_affine_anim_data.row(frame) =
          Vecxd::Map(added_reorg_affines.data(), added_reorg_affines.size())
              .transpose();
    }
    igl::writeDMAT(assets_path + "cheb_anim.dmat", affine_anim_data);
    igl::writeDMAT(assets_path + "cheb_added_anim.dmat",
                   added_affine_anim_data);
    frame = 0;
  }

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    handle_gui_input(gui);

    update_bone_rig(c_p, c_p_ref, anim_data.row(frame).transpose(), c_e);
    frame = (frame + 1) % anim_data.rows();

    aphys::Vecxd anim_frame_data = anim_data.row(frame);
    aphys::MatxXd affines =
        aphys::MatxXd::Map(anim_frame_data.data(), c_e.rows() * 3, 4);
    aphys::MatxXd reorg_affines(cW.rows() * 4, 3);
    aphys::MatxXd added_reorg_affines(added_cW.rows() * 4, 3);
    for (int i = 0; i < c_e.rows(); i++) {
      reorg_affines.block(i * 4, 0, 4, 3) =
          affines.block(i * 3, 0, 3, 4).transpose();
      added_reorg_affines.block(i * 4, 0, 4, 3) =
          affines.block(i * 3, 0, 3, 4).transpose();
    }
    added_reorg_affines.block((c_e.rows()) * 4, 0, 4, 3) =
        affines.block(2 * 3, 0, 3, 4).transpose();
    added_reorg_affines.block((c_e.rows() + 1) * 4, 0, 4, 3) =
        affines.block(2 * 3, 0, 3, 4).transpose();

    aphys::MatxXd verts = W * reorg_affines;
    // aphys::MatxXd verts = added_W * added_reorg_affines;
    construct_visual_tets(vtm.visual_verts, verts, tm.tets);
    set_color_mesh_data(visual_mesh, vtm.visual_verts.cast<float>(),
                        vtm.visual_faces, vtm.visual_colors);

    aphys::MatxXd c_verts = cW * reorg_affines;
    aphys::MatxXd added_c_verts = added_cW * added_reorg_affines;

    set_points_data(cheb_points, c_verts.cast<float>(), MatxXf());
    set_edges_data(cheb_edges, c_verts.cast<float>(), c_e, MatxXf());
    set_points_data(addeed_cheb_points, added_c_verts.cast<float>(), MatxXf());
    set_edges_data(added_cheb_edges, added_c_verts.cast<float>(), added_ce,
                   MatxXf());

    aphys::set_background_RGB({244, 244, 244});
    aphys::orbit_camera_control(window, scene.camera, 10.0, scene.delta_time);
    render_scene(scene);
    render_gui(gui);
    glfwSwapBuffers(window);
  }
  destroy_gui(gui);
  glfwTerminate();
}
