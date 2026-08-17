package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.linear.Matrix;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;

public class LeastSquare {

    private static final int U = 0;
    private static final int TEMP = 1;

    private static final int V = 0;
    private static final int W = 1;
    private static final int TEMP2 = 2;

    public static int solve(Matrix A, double[] b, double[] x0, int maxIter, float tol) {
        return solve(A, new CpuDoubleVector(b), new CpuDoubleVector(x0), maxIter, tol);
    }

    public static int solve(Matrix A, DoubleVector b, DoubleVector x0, int maxIter, float tol) {
        int m = A.rows();
        int n = A.cols();
        return solve(A, b, x0, maxIter, tol, new WorkingBuffer<>(2, () -> new CpuDoubleVector(m)),
                new WorkingBuffer<>(3, () -> new CpuDoubleVector(n))
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
    public static int solve(Matrix A, DoubleVector b, DoubleVector x, int maxIter, double tol, WorkingBuffer<DoubleVector> mBuffer, WorkingBuffer<DoubleVector> nBuffer) {

        int m = A.rows();
        int n = A.cols();


        if (b.size() != m)
            throw new IllegalArgumentException("b size != A.rows()");

        x.resize(n);

        DoubleVector u = mBuffer.get(U);
        DoubleVector temp = mBuffer.get(TEMP);

        DoubleVector v = nBuffer.get(V);
        DoubleVector w = nBuffer.get(W);
        DoubleVector temp2 = nBuffer.get(TEMP2);

        u.resize(m);
        temp.resize(m);

        v.resize(n);
        w.resize(n);
        temp2.resize(n);

        // u = b / ||b||

        u.copy(b);
        //System.out.println("b norm : "+ b.norm());
        double beta = u.norm();
        //System.out.println("beta = " + beta);

        if (beta == 0)
            return 0;

        u.scale(1.0 / beta);

        // v = Aᵀu
        A.transposeApply(u, v);

        double alpha = v.norm();
        //System.out.println("alpha = " + alpha);

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