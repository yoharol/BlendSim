#ifndef SIMTF_EXPORT_H_
#define SIMTF_EXPORT_H_

#include <ArmorerPhys/type.h>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#include "simTF/PD/pd.h"
#include "simTF/type.h"
#include "simTF/spline.h"

namespace aphys {

struct TrajectoryExporter {
  std::vector<MatxXd> frames;
  Matx3i faces;
  Eigen::MatrixXd weights;
  std::vector<double> tau;

  TrajectoryExporter(const SplineTrajectory& traj, const Matx4i& tets,
                     int sampling_N = 10);

  TrajectoryExporter(const SplineTrajectory& traj, const Matx3i& faces,
                     const MatxXd& weight_mat, int sampling_N = 10);

  void SaveAnimationToUSD(const std::string& filename);
};

void SaveFrameAnimatinoToUsd(SplineTrajectory& traj, const Matx4i& tets,
                             const std::string& filename, int fps = 24);

void SaveFrameAnimatinoToUsd(SplineTrajectory& traj, const Matx3i& faces,
                             const MatxXd& weight_mat,
                             const std::string& filename, int fps = 24);

}  // namespace aphys

#endif  // SIMTF_EXPORT_H_
