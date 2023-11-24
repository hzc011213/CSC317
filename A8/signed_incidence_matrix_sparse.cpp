#include "signed_incidence_matrix_sparse.h"
#include <vector>

void signed_incidence_matrix_sparse(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::SparseMatrix<double>  & A)
{
    std::vector<Eigen::Triplet<double> > ijv;

    for (int e = 0; e < E.rows(); e++) {
        ijv.emplace_back(i, E(e, 0), 1);
        ijv.emplace_back(i, E(e, 1), -1);

    }
    A.resize(E.rows(),n);
    A.setFromTriplets(ijv.begin(),ijv.end());
}
