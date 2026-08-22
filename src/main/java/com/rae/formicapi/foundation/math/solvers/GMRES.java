package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.linear.Matrix;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;

public class GMRES {

    private static final int RESIDUAL = 0;
    private static final int TEMP = 1;

    public static int solve(Matrix A, double[] b, double[] x0, int maxIter, float tol) {
        return solve(A, new CpuDoubleVector(b), new CpuDoubleVector(x0), maxIter, tol);
    }

    public static int solve(Matrix A, DoubleVector b, DoubleVector x0, int maxIter, float tol) {
        int m = A.rows();
        int n = A.cols();
        return solve(A, b, x0, maxIter, tol,
                new WorkingBuffer<>(2, () -> new CpuDoubleVector(n)),
                new WorkingBuffer<>(maxIter + 1, () -> new CpuDoubleVector(n))
        );
    }

    /**
     * Solve Ax = b using GMRES.
     *
     * The vector x is used as the initial guess and overwritten by the solution.
     *
     * The method minimizes the residual over the Krylov subspace
     *
     *     x_k = x_0 + K_k(A, r_0)
     *
     * where
     *
     *     K_k(A, r_0) = span(r_0, A*r_0, ..., A^(k-1)*r_0).
     *
     * mBuffer:
     *  RESIDUAL -> A.rows()
     *  TEMP     -> A.rows()
     *
     * nBuffer:
     *  V[0..maxIter] -> Krylov basis vectors
     */
    public static int solve(Matrix A, DoubleVector b, DoubleVector x, int maxIter, double tol, WorkingBuffer<DoubleVector> mBuffer, WorkingBuffer<DoubleVector> nBuffer) {

        int m = A.rows();
        int n = A.cols();

        if (m != n)
            throw new IllegalArgumentException("GMRES requires a square matrix");

        if (b.size() != m)
            throw new IllegalArgumentException("b size != A.rows()");

        x.resize(n);

        DoubleVector residual = mBuffer.get(RESIDUAL);
        DoubleVector temp = mBuffer.get(TEMP);

        residual.resize(n);
        temp.resize(n);

        DoubleVector v0 = nBuffer.get(0);
        v0.resize(n);

        // residual = b - A*x

        A.apply(x, temp);

        residual.copy(b);
        temp.scale(-1.0);
        residual.add(temp);

        double beta = residual.norm();

        if (beta == 0)
            return 0;

        v0.copy(residual);
        v0.scale(1.0 / beta);

        double[] h = new double[(maxIter + 1) * maxIter];
        double[] cs = new double[maxIter];
        double[] sn = new double[maxIter];
        double[] g = new double[maxIter + 1];

        g[0] = beta;

        for (int iter = 0; iter < maxIter; iter++) {
            DoubleVector v = nBuffer.get(iter);
            DoubleVector next = nBuffer.get(iter + 1);

            v.resize(n);
            next.resize(n);

            // next = A*v

            A.apply(v, next);

            // Arnoldi orthogonalization

            for (int j = 0; j <= iter; j++) {
                DoubleVector vj = nBuffer.get(j);

                double hij = next.dot(vj);

                h[j * maxIter + iter] = hij;

                next.axpy(-hij, vj);
            }

            double hNext = next.norm();

            h[(iter + 1) * maxIter + iter] = hNext;

            if (hNext != 0)
                next.scale(1.0 / hNext);

            // Apply previous Givens rotations

            for (int j = 0; j < iter; j++) {
                double h1 = h[j * maxIter + iter];
                double h2 = h[(j + 1) * maxIter + iter];

                double tempH = cs[j] * h1 + sn[j] * h2;

                h[(j + 1) * maxIter + iter] = -sn[j] * h1 + cs[j] * h2;
                h[j * maxIter + iter] = tempH;
            }

            // Generate new Givens rotation

            double h1 = h[iter * maxIter + iter];
            double h2 = h[(iter + 1) * maxIter + iter];

            double rho = Math.hypot(h1, h2);

            if (rho == 0)
                return iter;

            cs[iter] = h1 / rho;
            sn[iter] = h2 / rho;

            h[iter * maxIter + iter] = rho;
            h[(iter + 1) * maxIter + iter] = 0.0;

            // Apply Givens rotation to RHS

            double tempG = cs[iter] * g[iter] + sn[iter] * g[iter + 1];

            g[iter + 1] = -sn[iter] * g[iter] + cs[iter] * g[iter + 1];
            g[iter] = tempG;

            if (Math.abs(g[iter + 1]) < tol) {
                updateSolution(x, nBuffer, h, g, iter, maxIter);
                return iter + 1;
            }
        }

        updateSolution(x, nBuffer, h, g, maxIter - 1, maxIter);

        return maxIter;
    }

    private static void updateSolution(DoubleVector x, WorkingBuffer<DoubleVector> nBuffer, double[] h, double[] g, int iter, int maxIter) {

        int k = iter + 1;

        double[] y = new double[k];

        for (int i = k - 1; i >= 0; i--) {
            double value = g[i];

            for (int j = i + 1; j < k; j++)
                value -= h[i * maxIter + j] * y[j];

            y[i] = value / h[i * maxIter + i];
        }

        for (int i = 0; i < k; i++) {
            DoubleVector v = nBuffer.get(i);
            x.axpy(y[i], v);
        }
    }
}