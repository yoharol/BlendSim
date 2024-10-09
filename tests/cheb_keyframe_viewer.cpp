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
#include "simTF/lbs/model.h"

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

  TetMesh tm;
  VisualTetMesh vtm;
  Matx3f tm_color;

  std::string cnslash = "/";
  std::string assets_path = SIMTF_RESOURCES_PATH + cnslash + "cheb" + cnslash;

  igl::readMESH(assets_path + "cheb.mesh", tm.verts, tm.tets, tm.faces);

  extract_visual_tets_surfaces(tm.tets, tm.verts, vtm.visual_faces);
  // tm_color.resize(tm.verts.rows(), 3);
  // tm_color.setConstant(0.5f);

  LBSModel lbs_model(assets_path + "cheb_ref_points.dmat",  //
                     "",                                    //
                     assets_path + "cheb_bone_edges.dmat",  //
                     assets_path + "cheb_bbw_weight.dmat",  //
                     assets_path + "cheb_c_weight.dmat");

  LBSModel added_lbs_model(assets_path + "cheb_added_ref_points.dmat",  //
                           assets_path + "cheb_added_Points.dmat",      //
                           assets_path + "cheb_added_bone_edges.dmat",  //
                           assets_path + "cheb_added_bbw_weight.dmat",  //
                           assets_path + "cheb_added_c_weight.dmat");

  lbs_model.load_anim_data(assets_path + "cheb_selected_anim.dmat");
  lbs_model.initialize_lbs_weights(tm.verts);
  added_lbs_model.load_anim_data(assets_path + "cheb_added_selected_anim.dmat");
  added_lbs_model.initialize_lbs_weights(tm.verts);

  int n_frames = lbs_model.anim_frame_data.rows();
  SplineTrajectory lbs_traj;
  double total_time;
  {
    std::vector<MatxXd> key_frames(n_frames);
    for (int k = 0; k < n_frames; k++) {
      lbs_model.load_frame_data(k);
      key_frames[k] = lbs_model.T;
    }
    std::vector<int> segments(n_frames - 1);
    for (int i = 0; i < n_frames - 1; i++) {
      segments[i] = 1;
    }
    std::vector<double> time_interval(n_frames - 1);
    for (double& t : time_interval) {
      t = 0.2;
    }
    total_time = sum_vector(time_interval);
    set_keyframe_bezier_interpolation(key_frames, segments, time_interval,
                                      lbs_traj);
  }

  construct_visual_tets(vtm.visual_verts, tm.verts, tm.tets);
  DiffuseMesh visual_mesh = create_diffuse_mesh(material);
  add_render_func(scene, get_render_func(visual_mesh), true, true);
  construct_visual_tets(vtm.visual_verts, tm.verts, tm.tets);
  // construct_visual_tets_color(vtm.visual_colors, tm_color, tm.tets);
  set_mesh_data(visual_mesh, vtm.visual_verts.cast<float>(), vtm.visual_faces);

  DiffuseMesh traj_mesh = create_diffuse_mesh(material);
  add_render_func(scene, get_render_func(traj_mesh));

  lbs_model.initialize_render_objects(scene);
  added_lbs_model.initialize_render_objects(scene);

  glfwSwapInterval(1);

  int frame = 0;
  add_gui_func(gui, [&]() {
    ImGui::Text("Frame: %d", frame);
    bool add_frame = ImGui::Button("Add Frame");
    if (add_frame) {
      frame++;
      frame = frame % lbs_model.anim_frame_data.rows();
    }
  });

  double start_time = glfwGetTime();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    handle_gui_input(gui);

    double tau = (glfwGetTime() - start_time) / (4.0 / total_time);
    if (tau > total_time) {
      tau = 0.0;
      start_time = glfwGetTime();
    }

    added_lbs_model.load_frame_data(frame);
    // frame++;

    // aphys::MatxXd verts = lbs_model.lbs_weights * lbs_model.T;
    aphys::MatxXd verts = added_lbs_model.lbs_weights * added_lbs_model.T;
    construct_visual_tets(vtm.visual_verts, verts, tm.tets);
    set_mesh_data(visual_mesh, vtm.visual_verts.cast<float>(),
                  vtm.visual_faces);

    aphys::MatxXd sampled_T;
    aphys::MatxXd sampled_T_acce;
    sample_tau_on_trajectory(tau, lbs_traj, sampled_T, sampled_T_acce);
    verts = lbs_model.lbs_weights * sampled_T;
    construct_visual_tets(vtm.visual_verts, verts, tm.tets);
    set_mesh_data(traj_mesh, vtm.visual_verts.cast<float>(), vtm.visual_faces);

    aphys::set_background_RGB({244, 244, 244});
    aphys::orbit_camera_control(window, scene.camera, 10.0, scene.delta_time);
    render_scene(scene);
    render_gui(gui);
    glfwSwapBuffers(window);
  }
  destroy_gui(gui);
  glfwTerminate();
}
