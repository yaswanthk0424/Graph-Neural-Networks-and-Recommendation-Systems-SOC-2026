import numpy as np

def gram_schmidt(A):

    A = A.astype(float)

    m, n = A.shape

    Q = np.zeros((m, n))

    for k in range(n):

        # Take k-th column
        u = A[:, k]

        # Remove projections
        for j in range(k):

            proj_coeff = np.dot(Q[:, j], A[:, k]) / np.dot(Q[:, j], Q[:, j])

            u = u - proj_coeff * Q[:, j]

        # Normalize
        norm = np.linalg.norm(u)

        if norm == 0:
            raise ValueError("Vectors are linearly dependent.")

        Q[:, k] = u / norm

    return Q


# ---------------- TEST ---------------- #

A = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
])

Q = gram_schmidt(A)

print("Q:\n", Q)

print("\nQ^T Q:\n", Q.T @ Q)

print("\nIs orthonormal?")
print(np.allclose(Q.T @ Q, np.eye(Q.shape[1])))
