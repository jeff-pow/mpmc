from typing import Callable
import numpy as np
from scipy.linalg import eigh

rng = np.random.default_rng()
np.set_printoptions(precision=15)


def stochastic_trace(matrix: np.ndarray, dim: int, n: int):
    sum = 0
    for i in range(n):
        rademacher = rng.uniform(size=dim)
        rademacher = np.array([-1 if x < 0.5 else 1 for x in rademacher])
        sum += rademacher.T @ matrix @ rademacher
    return sum / n


def reorthagonalize(v: np.ndarray, other_vs: list[np.ndarray]):
    for other_v in other_vs:
        v -= v.dot(other_v) * other_v
        v /= np.linalg.norm(v)
    return v


def mtx_mult(matrix: np.ndarray, v: np.ndarray):
    result = [0] * len(v)
    for i in range(len(v)):
        for j in range(len(v)):
            result[i] += matrix[i][j] * vector[j]
    return result


def lanczos(matrix: np.ndarray, v: np.ndarray, m: int, do_reorthagonalize: bool = True):
    # https://en.wikipedia.org/wiki/Lanczos_algorithm
    v /= np.linalg.norm(v)
    w = matrix @ v
    print(w)
    print()
    print(mtx_mult(matrix, v))
    exit()
    # print(matrix)
    # print()
    # for a in w:
    #     print(a)
    w = [0.206877763152701,
         -0.206877763152701,
         -0.193915859621501,
         0.206877763152701,
         -0.206877763152701,
         -0.193915859621501]
    w = np.array(w)
    alpha = w.T @ v
    print(w)
    print(v)
    print(alpha)
    w = w - alpha * v
    v_last = np.copy(v)
    alphas = [alpha]
    betas = []
    vs = [np.copy(v)]
    for i in range(m - 1):
        beta = np.linalg.norm(w)
        print(beta)
        exit()
        if beta == 0:
            # avoid division by zero, create new random vector to start over
            print("HERE")
            exit(0)
            v = rng.uniform(size=v.shape[0])
            v = np.array([-1.0 if x < 0.5 else 1.0 for x in v])
            v = [1, -1, -1, 1, -1, -1]
            v /= np.linalg.norm(v)
            # reorthagonalize(v, vs)
        else:
            v = w / beta
        # if do_reorthagonalize:
            # reorthagonalize(v, vs)
        w = matrix @ v
        alpha = w.T @ v
        w = w - alpha * v - beta * v_last
        v_last = v
        alphas.append(alpha)
        betas.append(beta)
        print("alpha:", alpha)
        print("beta:", beta)
        # exit(0)
        vs.append(np.copy(v))
    return alphas, betas


def slq_lanczos(matrix: np.ndarray, num_iters: int, dim: int, lanczos_size: int, f: Callable):
    # https://epubs.siam.org/doi/10.1137/16M1104974
    sum = 0
    for i in range(num_iters):
        rademacher = rng.uniform(size=dim)
        rademacher = np.array([-1.0 if x < 0.5 else 1.0 for x in rademacher])
        rademacher = [1, -1, -1, 1, -1, -1]
        rademacher /= np.linalg.norm(rademacher)
        alphas, betas = lanczos(matrix, rademacher, lanczos_size, do_reorthagonalize=True)
        print(alphas)
        exit(0)
        tridiag = np.diag(alphas) + np.diag(betas, -1) + np.diag(betas, 1)
        vals, vecs = eigh(tridiag)
        for j in range(lanczos_size):
            sum += f(vals[j]) * vecs[0][j]**2
    return sum * dim / num_iters


def test_mpmc_mtx():
    d = 6  # dimensionality

    A = np.loadtxt('./two-ar.txt')

    eig_values, eig_vectors = eigh(A)  # computing diagonalization of A

    assert (eig_values >= 0).all()  # ensuring square root matrix exists
    sqrt_A = eig_vectors * np.sqrt(eig_values) @ np.linalg.inv(eig_vectors)  # calculate sqrt(A)

    assert np.linalg.norm(sqrt_A @ sqrt_A - A) < 1e-6  # assert sqrt(A)^2 == A
    assert stochastic_trace(A, d, 1000) - np.trace(A) < 1e-1  # assert traces are equal
    assert stochastic_trace(sqrt_A, d, 1000) - np.trace(sqrt_A) < 1e-1  # assert traces are equal
    print(stochastic_trace(A, d, 6))
    print(np.trace(sqrt_A))


def test_stochastic_trace():
    d = 3  # dimensionality

    while True:
        A = np.eye(d) + 0.1 * rng.random(d * d).reshape((d, d))  # random matrix with most energy on diagonals
        A = (A + A.T) / 2  # symmetrize

        eig_values, eig_vectors = eigh(A)  # computing diagonalization of A
        if (eig_values >= 0).all():
            break

    assert (eig_values >= 0).all()  # ensuring square root matrix exists
    sqrt_A = eig_vectors * np.sqrt(eig_values) @ np.linalg.inv(eig_vectors)  # calculate sqrt(A)

    assert np.linalg.norm(sqrt_A @ sqrt_A - A) < 1e-6  # assert sqrt(A)^2 == A
    assert stochastic_trace(A, d, 1000) - np.trace(A) < 1e-1  # assert traces are equal
    assert stochastic_trace(sqrt_A, d, 1000) - np.trace(sqrt_A) < 1e-1  # assert traces are equal


def test_orthogonalize():
    d = 20
    vs = [rng.uniform(low=-1.0, size=d) for _ in range(d)]

    for idx, v in enumerate(vs):
        vs[idx] /= np.linalg.norm(v)

    for idx, v in enumerate(vs):
        if idx == 0:
            continue
        vs[idx] = reorthagonalize(vs[idx], vs[:idx])
        assert np.abs(np.linalg.norm(vs[idx]) - 1) < 1e-12

    for idx1, v1 in enumerate(vs):
        for idx2, v2 in enumerate(vs):
            if idx1 <= idx2:
                continue
            assert np.dot(v1, v2) < 1e-10


def test_lanczos():
    d = 100

    A = np.diag(np.arange(d)) + 0.5 * rng.random(d * d).reshape((d, d))  # random matrix with most energy on diagonals
    A = (A + A.T) / 2  # symmetrize

    eig_values, eig_vectors = eigh(A)  # computing diagonalization of A

    rademacher = rng.uniform(size=d)
    rademacher = np.array([-1.0 if x < 0.5 else 1.0 for x in rademacher])
    rademacher /= np.linalg.norm(rademacher)

    alphas, betas = lanczos(A, rademacher, d)
    tridiag = np.diag(alphas) + np.diag(betas, -1) + np.diag(betas, 1)
    tri_eig_vals, tri_eig_vecs = eigh(tridiag)
    print(len(alphas))
    print(len(eig_values))
    print(len(tri_eig_vals))

    for eig1, eig2 in zip(np.sort(eig_values), np.sort(tri_eig_vals)):
        assert eig1 - eig2 < 1e-1


def test_slq_lanczos_small():
    d = 50  # dimensionality
    while True:
        A = np.diag(np.arange(d)) + 1.0 * rng.random(d * d).reshape((d, d))  # random matrix with most energy on diagonals
        A = (A + A.T) / 2  # symmetrize

        eig_values, eig_vectors = eigh(A)  # computing diagonalization of A
        if (eig_values >= 0).all():
            break

    assert (eig_values >= 0).all()  # ensuring square root matrix exists
    sqrt_A = eig_vectors * np.sqrt(eig_values) @ np.linalg.inv(eig_vectors)  # calculate sqrt(A)

    assert np.trace(sqrt_A) - slq_lanczos(A, 100, d, d, np.sqrt) < 1e-4


def test():
    d = 6  # dimensionality
    A = np.loadtxt('./two-ar.txt')

    eig_values, eig_vectors = eigh(A)  # computing diagonalization of A

    assert (eig_values >= 0).all()  # ensuring square root matrix exists
    sqrt_A = eig_vectors * np.sqrt(eig_values) @ np.linalg.inv(eig_vectors)  # calculate sqrt(A)

    print(np.trace(sqrt_A))

    # print("30 50", slq_lanczos(A, 30, d, 50, np.sqrt))
    print("50 50", slq_lanczos(A, 6, d, d, np.sqrt))


def test_slq_lanczos_large():
    d = 1875  # dimensionality
    while True:
        A = np.diag(np.arange(d)) + 1.0 * rng.random(d * d).reshape((d, d))  # random matrix with most energy on diagonals
        A = (A + A.T) / 2  # symmetrize

        eig_values, eig_vectors = eigh(A)  # computing diagonalization of A
        if (eig_values >= 0).all():
            break

    assert (eig_values >= 0).all()  # ensuring square root matrix exists
    sqrt_A = eig_vectors * np.sqrt(eig_values) @ np.linalg.inv(eig_vectors)  # calculate sqrt(A)

    print(np.trace(sqrt_A))

    # print("30 50", slq_lanczos(A, 30, d, 50, np.sqrt))
    print("50 50", slq_lanczos(A, 50, d, 50, np.sqrt))
    # print("80 50", slq_lanczos(A, 80, d, 50, np.sqrt))
    # print("100 50", slq_lanczos(A, 100, d, 50, np.sqrt))

    # print("30 100", slq_lanczos(A, 30, d, 100, np.sqrt))
    # print("50 100", slq_lanczos(A, 50, d, 100, np.sqrt))
    # print("80 100", slq_lanczos(A, 80, d, 100, np.sqrt))
    # print("100 100", slq_lanczos(A, 100, d, 100, np.sqrt))

    # assert np.trace(sqrt_A) - slq_lanczos(A, 100, d, d//2, np.sqrt) < 1e-4


if __name__ == '__main__':
    # test_stochastic_trace()
    # test_orthogonalize()
    # test_lanczos()
    # test_slq_lanczos_small()
    # test_mpmc_mtx()
    test()
