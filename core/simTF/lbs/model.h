#ifndef SIMTF_LBS_MODEL_H_
#define SIMTF_LBS_MODEL_H_

#include "ArmorerPhys/type.h"
#include "ArmorerPhys/RenderCore.h"

namespace aphys {

struct LBSModel {
  int n_controls;
  int n_points;
  int n_edges;

  aphys::MatxXd cp_ref;
  aphys::MatxXd cp;
  aphys::Vecxi P;
  aphys::Matx2i BE;
  aphys::MatxXd bbw_weights;
  aphys::MatxXd cW;

  aphys::MatxXd T;
  SparseMatd lbs_weights;
  SparseMatd lbs_weights_ext;

  aphys::MatxXd anim_frame_data;

  Points control_points;
  Edges bone_edges;

  LBSModel(std::string cp_path, std::string P_path, std::string BE_path,
           std::string bbw_weights_path, std::string cW_path);

  void load_anim_data(std::string anim_data_path);

  void load_frame_data(int frame);

  void load_frame_affine(const aphys::MatxXd& frame_T);

  void initialize_lbs_weights(MatxXd& verts_ref);

  void initialize_render_objects(Scene& scene);
};

struct LBSModel2D {
  int n_controls;
  int n_points;
  int n_edges;

  aphys::MatxXd cp_ref;
  aphys::MatxXd cp;
  aphys::Vecxi P;
  aphys::Matx2i BE;
  aphys::MatxXd weights;
  Eigen::MatrixXd cW;
  aphys::MatxXd T;

  SparseMatd lbs_weights;
  SparseMatd lbs_weights_ext;

  aphys::MatxXd anim_frame_data;

  Points control_points;
  Edges bone_edges;

  LBSModel2D(MatxXd& v_p, MatxXd& cp_ref, Vecxi& P, Matx2i& BE,
             MatxXd& weights);

  void set_transform(const int idx, const Vec2d translate, const double angle);

  // void initialize_render_objects(Scene& scene);
};

}  // namespace aphys

#endif  // SIMTF_LBS_MODEL_H_
