#include "signed_incidence_matrix_dense.h"

void signed_incidence_matrix_dense(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::MatrixXd & A) {
    A = Eigen::MatrixXd::Zero(E.rows(), n);

    // Iterate over each edge
    for (int e = 0; e < E.rows(); e++) {
        int v1 = E(e, 0); // Start vertex of the edge
        int v2 = E(e, 1); // End vertex of the edge

        // Set the corresponding entries in the matrix
        A(e, v1)++;
        A(e, v2)--;
    }
}