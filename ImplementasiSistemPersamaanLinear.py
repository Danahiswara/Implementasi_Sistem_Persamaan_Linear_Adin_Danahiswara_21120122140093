class LinearSolver:
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def solve_inverse_matrix(self):
        n = len(self.A)
        A_inv = self.inverse_matrix(n)
        x = self.multiply_matrix_vector(A_inv, self.b)
        return x

    def inverse_matrix(self, n):
        identity = [[0]*n for _ in range(n)]
        for i in range(n):
            identity[i][i] = 1

        for i in range(n):
            factor = 1 / self.A[i][i]
            for j in range(n):
                self.A[i][j] *= factor
                identity[i][j] *= factor

            for k in range(n):
                if k == i:
                    continue
                factor = -self.A[k][i]
                for j in range(n):
                    self.A[k][j] += factor * self.A[i][j]
                    identity[k][j] += factor * identity[i][j]

        return identity

    def multiply_matrix_vector(self, A, v):
        n = len(A)
        m = len(v)
        result = [0]*n
        for i in range(n):
            for j in range(m):
                result[i] += A[i][j] * v[j]
        return result

    def solve_gauss(self):
        n = len(self.A)
        for i in range(n):
            for j in range(i+1, n):
                ratio = self.A[j][i]/self.A[i][i]
                for k in range(n):
                    self.A[j][k] -= ratio * self.A[i][k]
                self.b[j] -= ratio * self.b[i]
        x = [0]*n
        for i in range(n-1, -1, -1):
            x[i] = self.b[i]/self.A[i][i]
            for j in range(i-1, -1, -1):
                self.b[j] -= self.A[j][i]*x[i]
        return x

    def solve_crout(self):
        n = len(self.A)
        L = [[0]*n for _ in range(n)]
        U = [[0]*n for _ in range(n)]
        for i in range(n):
            U[i][i] = 1
        for i in range(n):
            L[i][0] = self.A[i][0]
        for j in range(n):
            U[0][j] = self.A[0][j] / L[0][0]
        for i in range(1, n):
            for j in range(i, n):
                sum = 0
                for k in range(i):
                    sum += L[i][k] * U[k][j]
                L[i][j] = self.A[i][j] - sum
            for j in range(i+1, n):
                sum = 0
                for k in range(i):
                    sum += L[j][k] * U[k][i]
                U[j][i] = (self.A[j][i] - sum) / L[i][i]
        y = [0]*n
        x = [0]*n
        for i in range(n):
            sum = 0
            for k in range(i):
                sum += L[i][k] * y[k]
            y[i] = (self.b[i] - sum) / L[i][i]
        for i in range(n-1, -1, -1):
            sum = 0
            for k in range(i+1, n):
                sum += U[i][k] * x[k]
            x[i] = (y[i] - sum) / U[i][i]
        return x

# Testing code
A = [[2, -1, 1],
     [3, 3, 9],
     [3, 3, 5]]
b = [2, 6, 10]

solver = LinearSolver(A, b)

print("Inverse Matrix Method:")
print(solver.solve_inverse_matrix())

print("\nGauss Elimination Method:")
print(solver.solve_gauss())

print("\nCrout Decomposition Method:")
print(solver.solve_crout())
