#include "fast_mass_springs_precomputation_sparse.h"
#include "signed_incidence_matrix_sparse.h"
#include <vector>

#define w 1e10

bool fast_mass_springs_precomputation_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::SparseMatrix<double>  & M,
  Eigen::SparseMatrix<double>  & A,
  Eigen::SparseMatrix<double>  & C,
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization)
{
    // Calculate list of edge lengths r
    r.resize(E.rows());
    for (int i = 0; i < E.rows(); i++) {
        auto v1 = V.row(E(i, 0));
        auto v2 = V.row(E(i, 1));
        r[i] = (v1 - v2).norm();
    }

    // Construct mass matrix M
    std::vector<Eigen::Triplet<double>> ijvM;
    for (int i = 0; i < V.rows(); i++) {
        ijvM.emplace_back(i, i, m[i]);
    }
    M.resize(V.rows(), V.rows());
    M.setFromTriplets(ijvM.begin(), ijvM.end());

    // Construct the signed incidence matrix A
    signed_incidence_matrix_sparse(V.rows(), E, A);

    // Construct the selection matrix for pinned vertices C
    std::vector<Eigen::Triplet<double>> ijvC;
    for (int i = 0; i < b.size(); i++) {
        ijvC.emplace_back(i, b[i], 1);
    }
    C.resize(b.size(), V.rows());
    C.setFromTriplets(ijvC.begin(), ijvC.end());

    // Assemble matrix Q
    Eigen::SparseMatrix<double> Q = k * A.transpose() * A + M / pow(delta_t, 2) + w * C.transpose() * C;

    prefactorization.compute(Q);
    return prefactorization.info() != Eigen::NumericalIssue;

}
