#ifndef SIMTF_TYPE_H_
#define SIMTF_TYPE_H_

#include <ArmorerPhys/type.h>
#include <ArmorerPhys/glmath.h>
#include <ArmorerPhys/tet.h>
#include <functional>

namespace aphys {

// using Energy_Jacobian_Func = std::function<void>(MatxXd& v_p, Vecxd&
// jacobian);
typedef std::function<void(MatxXd&, Vecxd&)> Energy_Jacobian_Func;
// using Energy_Hessian_Func = std::function<void>(MatxXd& v_p, MatxXd&
// hessian);
typedef std::function<void(MatxXd&, SparseMatd&)> Energy_Hessian_Func;

typedef std::function<void(MatxXd&, Vecxd&)> Local_Step_Func;

inline void add_sparse_block(SparseMatd& dst, SparseMatd& src, int row,
                             int col) {
  for_each_nonzero(src, [&](SparseMatdIter& it) {
    dst.coeffRef(row + it.row(), col + it.col()) += it.value();
  });
}

template <int dim>
void extendSparseMatrixByKronecker(const SparseMatd& input_matrix,
                                   SparseMatd& extended_matrix) {
  int rows = input_matrix.rows();
  int cols = input_matrix.cols();
  std::vector<Tripletd> triplets;
  triplets.reserve(input_matrix.nonZeros() * dim);

  for (int k = 0; k < input_matrix.outerSize(); ++k) {
    for (SparseMatd::InnerIterator it(input_matrix, k); it; ++it) {
      for (int d = 0; d < dim; d++) {
        triplets.push_back(
            Tripletd(it.row() * dim + d, it.col() * dim + d, it.value()));
      }
    }
  }
  extended_matrix.resize(rows * dim, cols * dim);
  extended_matrix.setFromTriplets(triplets.begin(), triplets.end());
}

}  // namespace aphys

#endif  // SIMTF_TYPE_H_
