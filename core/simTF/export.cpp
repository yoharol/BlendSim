#include "simTF/export.h"

#include <Eigen/Core>
#include <vector>
#include <cmath>
#include <iostream>
#include <ArmorerPhys/tet.h>
#include <igl/writeDMAT.h>

#ifdef USD_EXPORTATION
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/pointBased.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/blendShape.h>
#include <pxr/usd/usdSkel/animation.h>
#include <pxr/usd/usdSkel/bindingAPI.h>
#include <pxr/usd/usdSkel/blendShapeQuery.h>
#include <pxr/usd/usdSkel/blendShape.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdSkel/utils.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#endif  // USD_EXPORTATION

#include "simTF/type.h"
#include "simTF/spline.h"

namespace aphys {

TrajectoryExporter::TrajectoryExporter(const SplineTrajectory& traj,
                                       const Matx4i& tets, int sampling_N) {
#ifdef USD_EXPORTATION
  int n_keyframes = traj.n_keyframes;
  int n_verts = traj.n_verts;

  VisualTetMesh vtm;
  extract_visual_tets_surfaces(tets, traj.keyframes[0].pos, vtm.visual_faces);

  faces = vtm.visual_faces;

  int n_intervals = n_keyframes - 1;
  int n_blend_shapes = n_intervals * 3 + 1;
  tau.resize(n_intervals * sampling_N + 1);
  weights.resize(n_intervals * sampling_N + 1, n_blend_shapes);
  weights.setZero();

  int idx = 0;

  for (int i = 0; i < n_intervals; i++) {
    double currT = traj.T_between[i];
    if (i == 0) {
      construct_visual_tets(vtm.visual_verts, traj.keyframes[i].pos, tets);
      frames.push_back(vtm.visual_verts);
      idx++;
    }
    MatxXd bs1 = traj.keyframes[i].pos + currT * traj.keyframes[i].vel;
    construct_visual_tets(vtm.visual_verts, bs1, tets);
    frames.push_back(vtm.visual_verts);
    idx++;
    MatxXd bs2 = traj.keyframes[i + 1].pos - currT * traj.keyframes[i + 1].vel;
    construct_visual_tets(vtm.visual_verts, bs2, tets);
    frames.push_back(vtm.visual_verts);
    idx++;

    construct_visual_tets(vtm.visual_verts, traj.keyframes[i + 1].pos, tets);
    frames.push_back(vtm.visual_verts);
    idx++;

    int id0 = idx - 4;
    int id1 = idx - 3;
    int id2 = idx - 2;
    int id3 = idx - 1;

    int maxj = sampling_N;
    if (i == n_intervals - 1) {
      maxj = sampling_N + 1;
    }

    for (int j = 0; j < maxj; j++) {
      int fi = i * sampling_N + j;
      double t = (double)j / (double)sampling_N;

      double coe0 = pow(1 - t, 3);
      double coe1 = 3 * pow(1 - t, 2) * t;
      double coe2 = 3 * (1 - t) * pow(t, 2);
      double coe3 = pow(t, 3);

      double time_stamp = traj.tau_stamp[i] + t * currT;
      tau[fi] = time_stamp;

      weights(fi, id0) = coe0;
      weights(fi, id1) = coe1;
      weights(fi, id2) = coe2;
      weights(fi, id3) = coe3;
    }
  }

  std::cout << "n_frames: " << frames.size() << std::endl;
  std::cout << "n_faces: " << faces.rows() << std::endl;
  std::cout << "n_weights: " << weights.rows() << " " << weights.cols()
            << std::endl;
  std::cout << "n_tau: " << tau.size() << std::endl;
  std::cout << weights << std::endl;
  for (int i = 0; i < tau.size(); i++) {
    std::cout << tau[i] << " ";
  }
  std::cout << std::endl;
#endif  // USD_EXPORTATION
}

TrajectoryExporter::TrajectoryExporter(const SplineTrajectory& traj,
                                       const Matx3i& faces,
                                       const MatxXd& weight_mat, int sampling_N)
    : faces(faces) {
#ifdef USD_EXPORTATION
  int n_keyframes = traj.n_keyframes;
  int n_verts = weight_mat.rows();

  int n_intervals = n_keyframes - 1;
  int n_blend_shapes = n_intervals * 3 + 1;
  tau.resize(n_intervals * sampling_N + 1);
  weights.resize(n_intervals * sampling_N + 1, n_blend_shapes);
  weights.setZero();

  int idx = 0;

  for (int i = 0; i < n_intervals; i++) {
    double currT = traj.T_between[i];
    if (i == 0) {
      frames.push_back(weight_mat * traj.keyframes[i].pos);
      idx++;
    }
    MatxXd bs1 = traj.keyframes[i].pos + currT * traj.keyframes[i].vel;
    frames.push_back(weight_mat * bs1);
    idx++;
    MatxXd bs2 = traj.keyframes[i + 1].pos - currT * traj.keyframes[i + 1].vel;
    frames.push_back(weight_mat * bs2);
    idx++;

    frames.push_back(weight_mat * traj.keyframes[i + 1].pos);
    idx++;

    int id0 = idx - 4;
    int id1 = idx - 3;
    int id2 = idx - 2;
    int id3 = idx - 1;

    int maxj = sampling_N;
    if (i == n_intervals - 1) {
      maxj = sampling_N + 1;
    }

    for (int j = 0; j < maxj; j++) {
      int fi = i * sampling_N + j;
      double t = (double)j / (double)sampling_N;

      double coe0 = pow(1 - t, 3);
      double coe1 = 3 * pow(1 - t, 2) * t;
      double coe2 = 3 * (1 - t) * pow(t, 2);
      double coe3 = pow(t, 3);

      double time_stamp = traj.tau_stamp[i] + t * currT;
      tau[fi] = time_stamp;

      weights(fi, id0) = coe0;
      weights(fi, id1) = coe1;
      weights(fi, id2) = coe2;
      weights(fi, id3) = coe3;
    }
  }
  std::cout << std::endl;
#endif  // USD_EXPORTATION
}

void TrajectoryExporter::SaveAnimationToUSD(const std::string& filename) {
#ifdef USD_EXPORTATION
  using namespace pxr;

  double total_time = tau.back();
  int total_frames = 60 * total_time;

  if (frames.empty()) {
    std::cerr << "Frames vector is empty." << std::endl;
    return;
  }

  // Verify that all frames have the same number of vertices
  size_t numVertices = frames[0].rows();
  for (const auto& frame : frames) {
    if (frame.rows() != numVertices) {
      std::cerr << "All frames must have the same number of vertices."
                << std::endl;
      return;
    }
  }

  // Verify that weights and T have the same number of frames
  if (weights.rows() != tau.size()) {
    std::cerr
        << "Weights and time vector T must have the same number of frames."
        << std::endl;
    return;
  }

  // Verify that the number of blend shapes matches the frames
  size_t numBlendShapes = frames.size() - 1;  // Exclude base mesh
  if (weights.cols() - 1 != numBlendShapes) {
    std::cerr << "The number of blend shapes in weights does not match frames."
              << std::endl;
    return;
  }

  // Create a new USD stage
  UsdStageRefPtr stage = UsdStage::CreateNew(filename);
  if (!stage) {
    std::cerr << "Failed to create USD stage." << std::endl;
    return;
  }

  stage->SetStartTimeCode(0.0);
  stage->SetEndTimeCode(total_frames);
  stage->SetFramesPerSecond(60.0);

  // Define the SkelRoot
  UsdSkelRoot skelRoot = UsdSkelRoot::Define(stage, SdfPath("/SkelRoot"));
  UsdSkelSkeleton skel =
      UsdSkelSkeleton::Define(stage, SdfPath("/SkelRoot/Skeleton"));
  // Define the mesh under the SkelRoot
  UsdGeomMesh mesh = UsdGeomMesh::Define(stage, SdfPath("/SkelRoot/MyMesh"));

  // Set the points for the base mesh
  VtVec3fArray points(numVertices);
  for (size_t i = 0; i < numVertices; ++i) {
    points[i] = GfVec3f(static_cast<float>(frames[0](i, 0)),
                        static_cast<float>(frames[0](i, 1)),
                        static_cast<float>(frames[0](i, 2)));
  }
  mesh.GetPointsAttr().Set(points);

  // Set the face vertex counts and indices
  size_t numFaces = faces.rows();
  VtIntArray faceVertexCounts(numFaces, 3);  // Triangles
  VtIntArray faceVertexIndices(numFaces * 3);
  for (size_t i = 0; i < numFaces; ++i) {
    faceVertexIndices[i * 3 + 0] = faces(i, 0);
    faceVertexIndices[i * 3 + 1] = faces(i, 1);
    faceVertexIndices[i * 3 + 2] = faces(i, 2);
  }
  mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts);
  mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices);

  // Define blend shapes
  std::vector<UsdSkelBlendShape> blendShapes;
  VtTokenArray blendShapeNames;
  std::vector<SdfPath> blendShapePaths;

  for (size_t i = 1; i < frames.size(); ++i) {
    std::string blendShapeName = "BlendShape_" + std::to_string(i);
    SdfPath blendShapePath = SdfPath("/SkelRoot/BlendShapes/" + blendShapeName);
    UsdSkelBlendShape blendShape =
        UsdSkelBlendShape::Define(stage, blendShapePath);

    // Compute delta positions
    VtVec3fArray deltaPoints(numVertices);
    for (size_t j = 0; j < numVertices; ++j) {
      deltaPoints[j] =
          GfVec3f(static_cast<float>(frames[i](j, 0) - frames[0](j, 0)),
                  static_cast<float>(frames[i](j, 1) - frames[0](j, 1)),
                  static_cast<float>(frames[i](j, 2) - frames[0](j, 2)));
    }

    // Set the offsets (delta positions)
    blendShape.GetOffsetsAttr().Set(deltaPoints);

    // Optionally set other attributes like normals, if needed

    blendShapes.push_back(blendShape);
    blendShapeNames.push_back(TfToken(blendShapeName));
    blendShapePaths.push_back(blendShapePath);
  }

  UsdSkelBindingAPI mesh_binding = UsdSkelBindingAPI::Apply(mesh.GetPrim());
  mesh_binding.CreateSkeletonRel().SetTargets({skel.GetPath()});
  mesh_binding.GetBlendShapesAttr().Set(blendShapeNames);
  mesh_binding.GetBlendShapeTargetsRel().SetTargets(blendShapePaths);

  // Create an Animation
  UsdSkelAnimation skelAnimation =
      UsdSkelAnimation::Define(stage, SdfPath("/SkelRoot/Anim/MyAnim"));
  skelAnimation.GetBlendShapesAttr().Set(blendShapeNames);
  UsdAttribute weightAttr = skelAnimation.GetBlendShapeWeightsAttr();
  std::vector<double> timeSamples = tau;
  for (size_t i = 0; i < timeSamples.size(); ++i) {
    VtFloatArray weightSample(numBlendShapes);
    for (size_t j = 0; j < numBlendShapes; ++j) {
      weightSample[j] = static_cast<float>(weights(i, j + 1));
    }
    weightAttr.Set(weightSample, timeSamples[i] * 60.0);
  }

  weightAttr.SetMetadata(TfToken("interpolation"), VtValue(TfToken("bezier")));

  UsdSkelBindingAPI skel_binding = UsdSkelBindingAPI::Apply(skel.GetPrim());
  skel_binding.CreateAnimationSourceRel().SetTargets({skelAnimation.GetPath()});

  // Save the USD stage
  stage->GetRootLayer()->Save();

  std::cout << "Successfully saved USD file: " << filename << std::endl;
#endif  // USD_EXPORTATION
}

void SaveFrameAnimatinoToUsd(SplineTrajectory& traj, const Matx4i& tets,
                             const std::string& filename, int fps) {
#ifdef USD_EXPORTATION
  using namespace pxr;

  double max_tau = traj.tau_stamp.back();

  VisualTetMesh vtm;
  extract_visual_tets_surfaces(tets, traj.keyframes[0].pos, vtm.visual_faces);

  Matx3i faces = vtm.visual_faces;

  int n_frames = max_tau * fps + 1;

  UsdStageRefPtr stage = UsdStage::CreateNew(filename);
  if (!stage) {
    std::cerr << "Failed to create USD stage." << std::endl;
    return;
  }

  stage->SetStartTimeCode(0.0);
  stage->SetEndTimeCode(n_frames);
  stage->SetFramesPerSecond(24.0);

  UsdGeomMesh mesh = UsdGeomMesh::Define(stage, SdfPath("/root/MyMesh"));
  size_t numFaces = faces.rows();
  VtArray<int> faceVertexCounts(numFaces, 3);  // Triangles
  VtArray<int> faceVertexIndices;
  faceVertexIndices.reserve(numFaces * 3);
  for (size_t i = 0; i < numFaces; ++i) {
    faceVertexIndices.push_back(faces(i, 0));
    faceVertexIndices.push_back(faces(i, 1));
    faceVertexIndices.push_back(faces(i, 2));
  }
  mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts);
  mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices);

  UsdAttribute pointsAttr = mesh.CreatePointsAttr();

  for (int i = 0; i < n_frames; i++) {
    double tau = i / (double)fps;
    MatxXd v_p, v_acce, v_vel;
    sample_tau_on_trajectory(tau, traj, v_p, v_acce);
    construct_visual_tets(vtm.visual_verts, v_p, tets);
    MatxXd& verts = vtm.visual_verts;

    VtVec3fArray points(verts.rows());
    for (size_t j = 0; j < verts.rows(); ++j) {
      points[j] = GfVec3f(static_cast<float>(verts(j, 0)),
                          static_cast<float>(verts(j, 1)),
                          static_cast<float>(verts(j, 2)));
    }
    mesh.GetPointsAttr().Set(points, tau * fps);
  }

  stage->GetRootLayer()->Save();
  std::cout << "Successfully saved USD file: " << filename << std::endl;
#endif  // USD_EXPORTATION
}

void SaveFrameAnimatinoToUsd(SplineTrajectory& traj, const Matx3i& faces,
                             const MatxXd& weight_mat,
                             const std::string& filename, int fps) {
#ifdef USD_EXPORTATION
  using namespace pxr;

  double max_tau = traj.tau_stamp.back();

  int n_frames = max_tau * fps + 1;

  UsdStageRefPtr stage = UsdStage::CreateNew(filename);
  if (!stage) {
    std::cerr << "Failed to create USD stage." << std::endl;
    return;
  }

  stage->SetStartTimeCode(0.0);
  stage->SetEndTimeCode(n_frames);
  stage->SetFramesPerSecond(24.0);

  UsdGeomMesh mesh = UsdGeomMesh::Define(stage, SdfPath("/root/MyMesh"));
  size_t numFaces = faces.rows();
  VtArray<int> faceVertexCounts(numFaces, 3);  // Triangles
  VtArray<int> faceVertexIndices;
  faceVertexIndices.reserve(numFaces * 3);
  for (size_t i = 0; i < numFaces; ++i) {
    faceVertexIndices.push_back(faces(i, 0));
    faceVertexIndices.push_back(faces(i, 1));
    faceVertexIndices.push_back(faces(i, 2));
  }
  mesh.GetFaceVertexCountsAttr().Set(faceVertexCounts);
  mesh.GetFaceVertexIndicesAttr().Set(faceVertexIndices);

  UsdAttribute pointsAttr = mesh.CreatePointsAttr();

  for (int i = 0; i < n_frames; i++) {
    double tau = i / (double)fps;
    MatxXd v_p, v_acce, v_vel;
    sample_tau_on_trajectory(tau, traj, v_p, v_acce);
    MatxXd verts = weight_mat * v_p;

    VtVec3fArray points(verts.rows());
    for (size_t j = 0; j < verts.rows(); ++j) {
      points[j] = GfVec3f(static_cast<float>(verts(j, 0)),
                          static_cast<float>(verts(j, 1)),
                          static_cast<float>(verts(j, 2)));
    }
    mesh.GetPointsAttr().Set(points, tau * fps);
  }

  stage->GetRootLayer()->Save();
  std::cout << "Successfully saved USD file: " << filename << std::endl;
#endif  // USD_EXPORTATION
}

}  // namespace aphys
