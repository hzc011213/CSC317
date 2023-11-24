#include "fast_mass_springs_precomputation_dense.h"
#include "signed_incidence_matrix_dense.h"
#include <Eigen/Dense>

#define w 1e10


bool fast_mass_springs_precomputation_dense(
        const Eigen::MatrixXd &V,
        const Eigen::MatrixXi &E,
        const double k,
        const Eigen::VectorXd &m,
        const Eigen::VectorXi &b,
        const double delta_t,
        Eigen::VectorXd &r,
        Eigen::MatrixXd &M,
        Eigen::MatrixXd &A,
        Eigen::MatrixXd &C,
        Eigen::LLT <Eigen::MatrixXd> &prefactorization) {
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(V.rows(), V.rows());

    // Calculate list of edge lengths r
    r.resize(E.rows());
    for (int i = 0; i < E.rows(); i++) {
        auto v1 = V.row(E(i, 0));
        auto v2 = V.row(E(i, 1));
        r[i] = (v1 - v2).norm();
    }

    // Construct mass matrix M
    M = Eigen::MatrixXd::Zero(V.rows(), V.rows());
    for (int i = 0; i < V.rows(); i++) {
        M(i, i) = m[i];
    }

    // Construct the signed incidence matrix A
    signed_incidence_matrix_dense(V.rows(), E, A);

    // Construct the selection matrix for pinned vertices C
    C = Eigen::MatrixXd::Zero(b.size(), V.rows());
    for (int i = 0; i < b.size(); i++) {
        C(i, b[i]) = 1;
    }

    // Assemble matrix Q
    Q = k * A.transpose() * A + M / pow(delta_t, 2) + w * C.transpose() * C;

    prefactorization.compute(Q);
    return prefactorization.info() != Eigen::NumericalIssue;
}
