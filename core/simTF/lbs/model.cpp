#include "simTF/lbs/model.h"

#include <igl/readDMAT.h>
#include <igl/lbs_matrix.h>

namespace aphys {
LBSModel::LBSModel(std::string cp_path, std::string P_path, std::string BE_path,
                   std::string bbw_weights_path, std::string cW_path) {
  igl::readDMAT(cp_path, cp_ref);
  if (P_path != "") {
    igl::readDMAT(P_path, P);
  }
  if (BE_path != "") {
    igl::readDMAT(BE_path, BE);
  }
  igl::readDMAT(bbw_weights_path, bbw_weights);
  igl::readDMAT(cW_path, cW);

  n_points = P.size();
  n_edges = BE.rows();
  n_controls = n_points + n_edges;
  ;
  CheckError(cW.cols() == n_controls * 4, "LBSModel: cW size error");
  T.resize(n_controls * 4, 3);
  for (int i = 0; i < n_controls; i++) {
    T.block(i * 4, 0, 3, 3) = Mat3d::Identity();
  }
  cp = cW * T;

  control_points = create_points();
  bone_edges = create_edges();
  control_points.point_size = 5.0f;
}

void LBSModel::load_anim_data(std::string anim_data_path) {
  igl::readDMAT(anim_data_path, anim_frame_data);
  CheckError(anim_frame_data.cols() == n_controls * 3 * 4,
             "LBSModel: anim_frame_data size error");
}

void LBSModel::load_frame_data(int frame) {
  Vecxd frame_data = anim_frame_data.row(frame);
  T = MatxXd::Map(frame_data.data(), n_controls * 4, 3);
  cp = cW * T;

  set_points_data(control_points, cp.cast<float>(), MatxXf());
  set_edges_data(bone_edges, cp.cast<float>(), BE, MatxXf());
}

void LBSModel::load_frame_affine(const aphys::MatxXd& frame_T) {
  T = frame_T;
  cp = cW * T;

  set_points_data(control_points, cp.cast<float>(), MatxXf());
  set_edges_data(bone_edges, cp.cast<float>(), BE, MatxXf());
}

void LBSModel::initialize_lbs_weights(MatxXd& verts_ref) {
  Eigen::MatrixXd lw;
  igl::lbs_matrix(verts_ref, bbw_weights, lw);
  lbs_weights = lw.sparseView();
  int dim = 3;
  lbs_weights_ext.resize(lbs_weights.rows() * dim, lbs_weights.cols() * dim);
  for_each_nonzero(lbs_weights, [&](const SparseMatd::InnerIterator& it) {
    for (int d = 0; d < dim; d++)
      lbs_weights_ext.insert(it.row() * dim + d, it.col() * dim + d) =
          it.value();
  });
}

void LBSModel::initialize_render_objects(Scene& scene) {
  add_render_func(scene, get_render_func(control_points), false);
  add_render_func(scene, get_render_func(bone_edges), false);
}

}  // namespace aphys
