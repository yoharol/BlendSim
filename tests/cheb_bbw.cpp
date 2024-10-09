// bounded biharmonic weights

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <ArmorerPhys/RenderCore.h>

#include "igl/readOBJ.h"
#include "igl/readMESH.h"
#include "igl/readDMAT.h"
#include "igl/writeDMAT.h"
#include "igl/bbw.h"
#include "igl/writeDMAT.h"
#include "igl/boundary_conditions.h"

void lbs_control_matrix(const aphys::MatxXd C, const aphys::Vecxi& P,
                        const aphys::Matx2i& BE, aphys::MatxXd& W) {
  int n_p = P.size();
  int n_e = BE.rows();
  int dim = 3;

  int n_controls = n_p + n_e;

  aphys::Vecxi control_map(C.rows());
  for (int i = 0; i < n_p; i++) {
    control_map(P(i)) = i + n_e;
  }
  for (int i = 0; i < n_e; i++) {
    control_map(BE(i, 0)) = i;
    control_map(BE(i, 1)) = i;
  }

  W.resize(C.rows(), (dim + 1) * n_controls);
  W.setZero();
  for (int i = 0; i < C.rows(); i++) {
    aphys::RowVecxd homo_vec(dim + 1);
    homo_vec << C.row(i), 1.0;
    int cidx = control_map(i);
    W.block(i, cidx * (dim + 1), 1, dim + 1) = homo_vec;
  }
}

int main() {
  using namespace aphys;
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

  MatxXd c_p_ref;
  MatxXd c_p;
  Matx2i c_e;
  MatxXd anim_data;

  TetMesh tm;
  VisualTetMesh vtm;
  Matx3f tm_color;

  std::string cnslash = "/";
  std::string assets_path = SIMTF_RESOURCES_PATH + cnslash + "cheb" + cnslash;

  Box3d box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

  igl::readDMAT(assets_path + "cheb_ref_points.dmat", c_p_ref);
  igl::readDMAT(assets_path + "cheb_bone_edges.dmat", c_e);
  igl::readDMAT(assets_path + "cheb_affine_anim.dmat", anim_data);
  igl::readMESH(assets_path + "cheb.mesh", tm.verts, tm.tets, tm.faces);
  extract_visual_tets_surfaces(tm.tets, tm.verts, vtm.visual_faces);
  tm_color.resize(tm.verts.rows(), 3);
  tm_color.setConstant(0.5f);

  c_p = c_p_ref;

  Points cheb_points = create_points();
  cheb_points.point_size = 5.0f;
  Points added_points = create_points();
  added_points.point_size = 5.0f;
  Edges cheb_edges = create_edges();
  aphys::Edges box_edges = aphys::create_box_edges();
  box_edges.width = 0.8f;
  box_edges.color = {0, 0, 0};
  aphys::set_box_edges_data(box_edges, box);
  aphys::add_render_func(scene, aphys::get_render_func(box_edges));

  construct_visual_tets(vtm.visual_verts, tm.verts, tm.tets);
  ColorMesh visual_mesh = create_color_mesh(material);
  add_render_func(scene, get_render_func(visual_mesh));
  construct_visual_tets(vtm.visual_verts, tm.verts, tm.tets);
  construct_visual_tets_color(vtm.visual_colors, tm_color, tm.tets);
  set_color_mesh_data(visual_mesh, vtm.visual_verts.cast<float>(),
                      vtm.visual_faces, vtm.visual_colors);

  add_render_func(scene, get_render_func(cheb_points), false);
  add_render_func(scene, get_render_func(added_points), false);
  add_render_func(scene, get_render_func(cheb_edges), false);

  aphys::MatxXd weights;
  igl::BBWData bbw_data;
  bbw_data.active_set_params.max_iter = 8;
  bbw_data.verbosity = 2;

  for (int i = 0; i < c_p.rows(); i++) {
    c_p.row(i) =
        tm.verts.row(find_nearest_point(tm.verts, c_p_ref.row(i))).transpose();
  }
  aphys::MatxXd a_p(2, 3);
  a_p << 0.16, 0.75, 0.52, 0.8, 0.75, 0.48;
  a_p.row(0) =
      tm.verts.row(find_nearest_point(tm.verts, a_p.row(0))).transpose();
  a_p.row(1) =
      tm.verts.row(find_nearest_point(tm.verts, a_p.row(1))).transpose();
  aphys::MatxXd added_cp(c_p_ref.rows() + 2, 3);
  added_cp << c_p, a_p;

  Eigen::MatrixXd bc;
  Eigen::VectorXi b;
  Eigen::MatrixXd V = tm.verts;
  Eigen::MatrixXi T = tm.tets;
  Eigen::MatrixXd C = c_p;
  Eigen::MatrixXi BE = c_e;
  // Eigen::VectorXi P(C.rows());
  // P.setLinSpaced(C.rows(), 0, C.rows() - 1);
  // igl::boundary_conditions(V, T, C, P, Eigen::MatrixXi(), Eigen::MatrixXi(),
  //                          Eigen::MatrixXi(), b, bc);
  igl::boundary_conditions(V, T, C, Eigen::VectorXi(), BE, Eigen::MatrixXi(),
                           Eigen::MatrixXi(), b, bc);

  if (!igl::bbw(V, T, b, bc, bbw_data, weights)) {
    std::cout << "bbw failed" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "bbw success" << std::endl;
  for (int i = 0; i < weights.rows(); i++) {
    weights.row(i) /= weights.row(i).sum();
  }

  aphys::MatxXd cW;
  C = c_p_ref;
  lbs_control_matrix(C, Eigen::VectorXi(), BE, cW);
  igl::writeDMAT(assets_path + "cheb_c_weight.dmat", cW);
  igl::writeDMAT(assets_path + "cheb_bbw_weight.dmat", weights);

  Eigen::MatrixXd a_bc;
  Eigen::VectorXi a_b;
  Eigen::VectorXi P(2);
  igl::BBWData a_bbw_data;
  a_bbw_data.active_set_params.max_iter = 8;
  a_bbw_data.verbosity = 2;
  P << c_p_ref.rows(), c_p_ref.rows() + 1;
  C = added_cp;
  igl::boundary_conditions(V, T, C, P, BE, Eigen::MatrixXi(), Eigen::MatrixXi(),
                           a_b, a_bc);
  aphys::MatxXd added_weights;

  if (!igl::bbw(V, T, a_b, a_bc, a_bbw_data, added_weights)) {
    std::cout << "bbw failed" << std::endl;
    return EXIT_FAILURE;
  }

  {
    int n_edges = BE.rows();
    int n_points = P.size();
    Eigen::MatrixXd tmpW = added_weights;
    tmpW.leftCols(n_edges) = added_weights.rightCols(n_edges);
    tmpW.rightCols(n_points) = added_weights.leftCols(n_points);
    added_weights = tmpW;
  }
  std::cout << added_weights.rows() << "x" << added_weights.cols() << std::endl;

  for (int i = 0; i < added_weights.rows(); i++) {
    added_weights.row(i) /= added_weights.row(i).sum();
  }

  aphys::MatxXd added_cW;
  C.topRows(c_p_ref.rows()) = c_p_ref;
  lbs_control_matrix(C, P, BE, added_cW);
  igl::writeDMAT(assets_path + "cheb_added_c_weight.dmat", added_cW);
  igl::writeDMAT(assets_path + "cheb_added_ref_points.dmat", added_cp);
  igl::writeDMAT(assets_path + "cheb_added_bone_edges.dmat", BE);
  igl::writeDMAT(assets_path + "cheb_added_Points.dmat", P);
  igl::writeDMAT(assets_path + "cheb_added_bbw_weight.dmat", added_weights);

  // ! write a test matrix
  {
    std::cout << C.transpose() << std::endl;
    int n = P.size() + BE.rows();
    MatxXd T(n * 4, 3);
    T.setZero();
    for (int i = 0; i < n; i++) {
      T.block(i * 4, 0, 3, 3) = Mat3d::Identity();
    }
    std::cout << (added_cW * T).transpose() << std::endl;
  }

  auto set_weight_color = [&](int bone_idx) {
    for (int i = 0; i < tm.verts.rows(); i++) {
      aphys::Vecxf color =
          aphys::heat_rgb(added_weights(i, bone_idx), 0.0, 1.0).cast<float>();
      tm_color.row(i) = color.transpose();
    }
    construct_visual_tets_color(vtm.visual_colors, tm_color, tm.tets);
    set_color_mesh_data(visual_mesh, vtm.visual_verts.cast<float>(),
                        vtm.visual_faces, vtm.visual_colors);
  };

  int curr_bdx = 0;
  set_weight_color(0);

  add_gui_func(gui, [&]() {
    bool pressed_up = ImGui::Button("add");
    if (pressed_up) {
      curr_bdx = (curr_bdx + 1) % added_weights.cols();
      set_weight_color(curr_bdx);
    }
    bool pressed_down = ImGui::Button("sub");
    if (pressed_down) {
      curr_bdx = (curr_bdx - 1 + added_weights.cols()) % added_weights.cols();
      set_weight_color(curr_bdx);
    }
  });

  // ===================== main loop =====================
  glfwSwapInterval(1);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    aphys::handle_gui_input(gui);
    aphys::set_background_RGB({244, 244, 244});
    aphys::orbit_camera_control(window, scene.camera, 10.0, scene.delta_time);

    aphys::render_scene(scene);
    aphys::render_gui(gui);

    glfwSwapBuffers(window);
  }
  glfwTerminate();
}
