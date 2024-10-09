#include "simTF/renderer/renderer.h"

#include <iostream>

namespace aphys {

KeyframeEditingInterface::KeyframeEditingInterface(Scene& scene, Gui& gui,
                                                   SplineTrajectory& trajectory,
                                                   const TetMesh& tet_mesh,
                                                   DiffuseMaterial& material)
    : traj(trajectory), tet_faces(tet_mesh.tets) {
  int n_verts = trajectory.n_verts;
  extract_visual_tets_surfaces(tet_mesh.tets, tet_mesh.verts, visual_faces);
  extract_surface_from_tets(trajectory.n_verts, tet_mesh.tets, surface_faces);

  int n_keyframes = trajectory.n_keyframes;
  for (int k = 0; k < n_keyframes; k++) {
    const MatxXd& v_p = trajectory.keyframes[k].pos;

    MatxXd vvp;
    construct_visual_tets(vvp, v_p, tet_mesh.tets);
    visual_verts.push_back(vvp);
  }

  keyframe_mesh = create_diffuse_mesh(material);
  add_render_func(scene, get_render_func(keyframe_mesh));

  add_gui_func(gui, [n_keyframes, this]() {
    bool changed =
        ImGui::SliderInt("Keyframe Index", &curr_kdx, 0, n_keyframes - 1);
    if (changed) set_selection(curr_kdx);
    bool changed_editing =
        ImGui::Checkbox("Editing Keyframe", &editing_keyframe);
    if (changed_editing) setup_editing(curr_kdx, editing_keyframe);
  });

  add_gui_mouse_input_func(gui, [this, &scene, &gui, &trajectory, &tet_mesh]() {
    if (!editing_keyframe) return;
    MatxXd& curr_vp = trajectory.keyframes[curr_kdx].pos;
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      ImVec2 pos = ImGui::GetMousePos();
      float x = pos.x / gui.window_width;
      float y = 1.0 - pos.y / gui.window_height;
      aphys::Vec3f ray_originf, ray_dirf;
      aphys::camera_screen_to_raycast(scene.camera, x, y, ray_originf,
                                      ray_dirf);
      aphys::Vec3d ray_origin = ray_originf.cast<double>();
      aphys::Vec3d ray_dir = ray_dirf.cast<double>();
      edit_idx =
          aphys::raycast_face(curr_vp, surface_faces, ray_origin, ray_dir);
      ray_dis = (curr_vp.row(edit_idx).transpose() - ray_origin).dot(ray_dir);
      start_edit_pos = ray_origin + ray_dis * ray_dir;
      start_edit_ref_pos = curr_vp.row(edit_idx).transpose();
      dragging = true;
      if (edit_idx != -1) local_deform_solver.setupDeformer(curr_vp, edit_idx);
    }
    if (dragging && edit_idx != -1) {
      ImVec2 pos = ImGui::GetMousePos();
      float x = pos.x / gui.window_width;
      float y = 1.0 - pos.y / gui.window_height;
      aphys::Vec3f ray_originf, ray_dirf;
      aphys::camera_screen_to_raycast(scene.camera, x, y, ray_originf,
                                      ray_dirf);
      aphys::Vec3d ray_origin = ray_originf.cast<double>();
      aphys::Vec3d ray_dir = ray_dirf.cast<double>();
      Vec3d new_edit_pos = ray_origin + ray_dis * ray_dir;
      // curr_vp.row(edit_idx) =
      //     (start_edit_ref_pos + (new_edit_pos - start_edit_pos)).transpose();
      local_deform_solver.localStep(curr_vp);
      local_deform_solver.globalStep(curr_vp, new_edit_pos);
      trajectory.keyframes[curr_kdx].pos = curr_vp;

      MatxXd vvp;
      construct_visual_tets(vvp, curr_vp, tet_mesh.tets);
      visual_verts[curr_kdx] = vvp;
      set_mesh_data(keyframe_mesh, vvp.cast<float>(), visual_faces);
    }

    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      edit_idx = -1;
      dragging = false;
    }
  });

  set_selection(0);
}

void KeyframeEditingInterface::reset_mesh(MatxXd& v_p) {
  MatxXd vvp;
  construct_visual_tets(vvp, v_p, tet_faces);
  visual_verts[curr_kdx] = v_p;
  set_mesh_data(keyframe_mesh, visual_verts[curr_kdx].cast<float>(),
                visual_faces);
}

void KeyframeEditingInterface::set_selection(int idx, float alpha) {
  set_mesh_data(keyframe_mesh, visual_verts[idx].cast<float>(), visual_faces);
  keyframe_mesh.alpha = alpha;
}

void KeyframeEditingInterface::setup_editing(int idx, bool editing) {
  if (editing) {
    std::cout << "Editing keyframe " << idx << std::endl;

  } else {
    std::cout << "Finished editing keyframe " << idx << std::endl;
  }
}

void KeyframeEditingInterface::setup_deformation_params(
    const Matx4i& elements,  //
    const Vecxd& element_mass,
    const Vecxd& vert_mass,  //
    const MatxXd& external_force, const double stiffness_hydro,
    const double stiffness_devia, const double local_stiffness) {
  local_deform_solver.initializeParams(elements, element_mass, vert_mass,
                                       external_force, stiffness_hydro,
                                       stiffness_devia, local_stiffness);
}

/*
KeyframeEditingInterface2D::KeyframeEditingInterface2D(
    Scene& scene, Gui& gui, SplineTrajectory& trajectory, const Matx3i& faces,
    const Matx2i& edges, DiffuseMaterial& material)
    : traj(trajectory), faces(faces), edges(edges) {
  int n_verts = trajectory.n_verts;

  int n_keyframes = trajectory.n_keyframes;

  keyframe_edges = create_edges();

  add_render_func(scene, get_render_func(keyframe_edges));

  add_gui_func(gui, [n_keyframes, this]() {
    bool changed =
        ImGui::SliderInt("Keyframe Index", &curr_kdx, 0, n_keyframes - 1);
    if (changed) set_selection(curr_kdx);
    bool changed_editing =
        ImGui::Checkbox("Editing Keyframe", &editing_keyframe);
    if (changed_editing) setup_editing(curr_kdx, editing_keyframe);
  });

  add_gui_mouse_input_func(gui, [this, &scene, &gui, &trajectory, &faces,
                                 &edges]() {
    if (!editing_keyframe) return;
    MatxXd& curr_vp = trajectory.keyframes[curr_kdx].pos;
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      ImVec2 pos = ImGui::GetMousePos();
      float x = pos.x / gui.window_width;
      float y = 1.0 - pos.y / gui.window_height;
      camera2d_screen_to_world(scene.camera, x, y);
      start_edit_pos << x, y;
      edit_idx = find_nearest_point(curr_vp, start_edit_pos);
      start_edit_ref_pos = curr_vp.row(edit_idx).transpose();
      dragging = true;
      if (edit_idx != -1) local_deform_solver.setupDeformer(curr_vp, edit_idx);
    }
    if (dragging && edit_idx != -1) {
      ImVec2 pos = ImGui::GetMousePos();
      float x = pos.x / gui.window_width;
      float y = 1.0 - pos.y / gui.window_height;
      camera2d_screen_to_world(scene.camera, x, y);
      Vec2d new_edit_pos;
      new_edit_pos << x, y;
      local_deform_solver.localStep(curr_vp);
      local_deform_solver.globalStep(curr_vp, new_edit_pos);
      trajectory.keyframes[curr_kdx].pos = curr_vp;
      set_edges_data(keyframe_edges, curr_vp.cast<float>(), edges, MatxXf());

      MatxXd vvp;
      construct_visual_tets(vvp, curr_vp, tet_mesh.tets);
      visual_verts[curr_kdx] = vvp;
      set_mesh_data(keyframe_mesh, vvp.cast<float>(), visual_faces);
    }

    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
      edit_idx = -1;
      dragging = false;
    }
  });

  set_selection(0);
}
*/

void KeyframeEditingInterface::reset_keyframes(SplineTrajectory& trajectory) {
  int n_keyframes = trajectory.n_keyframes;
  for (int k = 0; k < n_keyframes; k++) {
    const MatxXd& v_p = trajectory.keyframes[k].pos;

    MatxXd vvp;
    construct_visual_tets(vvp, v_p, tet_faces);
    visual_verts[k] = vvp;
  }
  set_selection(0);
}

}  // namespace aphys
