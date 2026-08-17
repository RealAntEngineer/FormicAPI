package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuVector;
import com.rae.formicapi.foundation.math.operators.linear.Matrix;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;

public class LeastSquare {

    private static final int U = 0;
    private static final int TEMP = 1;

    private static final int V = 0;
    private static final int W = 1;
    private static final int TEMP2 = 2;

    public static int solve(Matrix A, double[] b, double[] x0, int maxIter, float tol) {
        return solve(A, new CpuVector(b), new CpuVector(x0), maxIter, tol);
    }

    public static int solve(Matrix A, Vector b, Vector x0, int maxIter, float tol) {
        int m = A.rows();
        int n = A.cols();
        return solve(A, b, maxIter, tol, x0, new WorkingBuffer(2, new CpuVector(m)),
                new WorkingBuffer(3, new CpuVector(n))
        );
    }

    /**
     * Solve min ||Ax-b|| using LSQR.
     *
     * Solution is written into x.
     *
     * mBuffer:
     *  U    -> A.rows()
     *  TEMP -> A.rows()
     *
     * nBuffer:
     *  V -> A.cols()
     *  W -> A.cols()
     */
    public static int solve(Matrix A, Vector b, int maxIter, double tol, Vector x, WorkingBuffer mBuffer, WorkingBuffer nBuffer) {

        int m = A.rows();
        int n = A.cols();


        if (b.size() != m)
            throw new IllegalArgumentException("b size != A.rows()");

        x.resize(n);

        Vector u = mBuffer.get(U);
        Vector temp = mBuffer.get(TEMP);

        Vector v = nBuffer.get(V);
        Vector w = nBuffer.get(W);
        Vector temp2 = nBuffer.get(TEMP2);

        u.resize(m);
        temp.resize(m);

        v.resize(n);
        w.resize(n);
        temp2.resize(n);

        // u = b / ||b||

        u.copy(b);

        double beta = u.norm();

        if (beta == 0)
            return 0;

        u.scale(1.0 / beta);

        // v = Aᵀu
        A.transposeApply(u, v);

        double alpha = v.norm();

        if (alpha == 0)
            return 0;

        v.scale(1.0 / alpha);
        w.copy(v);

        double phiBar = beta;
        double rhoBar = alpha;

        for (int iter = 0; iter < maxIter; iter++) {
            // u = A*v - alpha*u
            A.apply(v, temp);
            u.scale(-alpha);//TODO verify that this works
            u.add(temp);


            beta = u.norm();

            if (beta != 0)
                u.scale(1.0 / beta);

            // v = Aᵀu - beta*v
            A.transposeApply(u, temp2);
            v.scale(-beta);
            v.add(temp2);


            alpha = v.norm();

            if (alpha != 0)
                v.scale(1.0 / alpha);

            // rotation
            double rho = Math.sqrt(rhoBar * rhoBar + beta * beta);

            double c = rhoBar / rho;
            double s = beta / rho;
            double theta = s * alpha;

            rhoBar = -c * alpha;

            double phi = c * phiBar;
            phiBar = s * phiBar;

            // x += (phi/rho)w
            x.axpy(phi / rho, w);

            // w = v - (theta/rho)w
            w.scale(-theta / rho);
            w.add(v);

            if (Math.abs(phiBar) < tol)
                return iter + 1;
        }
        return maxIter;
    }
}