package com.rae.formicapi.math_tests.solvers;

import java.util.Arrays;

/**
 * Direct port of Matthias Muller's ("Ten Minute Physics") staggered-grid
 * (MAC) stable-fluids solver: Gauss-Seidel (SOR) pressure projection +
 * semi-Lagrangian advection. Original: Ten Minute Physics, MIT licensed,
 * www.matthiasMueller.info/tenMinutePhysics.
 *
 * <p>Deliberately a completely different numerical approach from the
 * {@code NewtonKrylov}-based tests in this package:
 * <ul>
 *   <li><b>Staggered (MAC) grid</b>, not collocated -- u lives on cell west
 *       faces, v on south faces, p at cell centers. This is the standard,
 *       textbook fix for the checkerboard pressure-velocity decoupling a
 *       collocated central-difference grid is prone to (see the
 *       cylinder/plate tests' notes on that risk).</li>
 *   <li><b>Semi-Lagrangian advection</b>: instead of discretizing
 *       {@code u.grad(u)} with finite differences -- exactly what forced the
 *       cell-Peclet-number fight in the other tests -- each velocity sample
 *       traces backward along the flow by {@code dt} and bilinearly
 *       interpolates the old field there. Unconditionally stable in
 *       {@code dt} regardless of velocity magnitude or grid spacing, no
 *       Peclet-number constraint at all, at the cost of real numerical
 *       dissipation from repeated interpolation.</li>
 *   <li><b>Gauss-Seidel/SOR pressure projection</b>: a fixed number of local
 *       relaxation sweeps, not a driven-to-tolerance Krylov solve.
 *       Divergence only gets approximately small, not exactly zero to a
 *       chosen tolerance -- more iterations (or a higher
 *       {@code overRelaxation}) buys more incompressibility, not an exact
 *       target.</li>
 * </ul>
 *
 * <p>Net effect: much simpler, much cheaper per step, and structurally
 * immune to the instabilities this package has spent a while fighting -- at
 * the cost of a visibly more dissipated, only-approximately-incompressible
 * result. Good for a fast qualitative check or a real-time visualization;
 * not a substitute for the Newton-Krylov tests if the point is validating
 * the tensor/solver machinery itself.
 *
 * <p>Flat-array indexing throughout is {@code i*numY + j} (x varies slower
 * than y), matching the original exactly -- kept as-is rather than
 * "improved" to minimize transcription risk against a known-good reference.
 */
public class StableFluidGrid {

    private static final int U_FIELD = 0;
    private static final int V_FIELD = 1;
    private static final int S_FIELD = 2; // smoke/dye

    public final int numX, numY, numCells;
    public final double h;
    public final double density;

    public final double[] u, v, newU, newV;
    public final double[] p;
    public final double[] s; // 1 = fluid, 0 = solid
    public final double[] m, newM; // smoke/dye, 1 = clear by convention here

    /**
     * @param density used only to scale the pressure field's units (cp = density*h/dt)
     * @param numX    interior cell count in x -- two ghost columns are added internally
     * @param numY    interior cell count in y -- two ghost rows are added internally
     * @param h       cell size
     */
    public StableFluidGrid(double density, int numX, int numY, double h) {
        this.density = density;
        this.numX = numX + 2;
        this.numY = numY + 2;
        this.numCells = this.numX * this.numY;
        this.h = h;

        u = new double[numCells];
        v = new double[numCells];
        newU = new double[numCells];
        newV = new double[numCells];
        p = new double[numCells];
        s = new double[numCells];
        m = new double[numCells];
        newM = new double[numCells];
        Arrays.fill(m, 1.0);
    }

    public void integrate(double dt, double gravity) {
        int n = numY;
        for (int i = 1; i < numX; i++) {
            for (int j = 1; j < numY - 1; j++) {
                if (s[i * n + j] != 0.0 && s[i * n + j - 1] != 0.0)
                    v[i * n + j] += gravity * dt;
            }
        }
    }

    public void solveIncompressibility(int numIters, double dt, double overRelaxation) {
        int n = numY;
        double cp = density * h / dt;

        for (int iter = 0; iter < numIters; iter++) {
            for (int i = 1; i < numX - 1; i++) {
                for (int j = 1; j < numY - 1; j++) {

                    if (s[i * n + j] == 0.0) continue;

                    double sx0 = s[(i - 1) * n + j];
                    double sx1 = s[(i + 1) * n + j];
                    double sy0 = s[i * n + j - 1];
                    double sy1 = s[i * n + j + 1];
                    double sSum = sx0 + sx1 + sy0 + sy1;
                    if (sSum == 0.0) continue;

                    double div = u[(i + 1) * n + j] - u[i * n + j]
                            + v[i * n + j + 1] - v[i * n + j];

                    double pCorr = -div / sSum;
                    pCorr *= overRelaxation;
                    p[i * n + j] += cp * pCorr;

                    u[i * n + j] -= sx0 * pCorr;
                    u[(i + 1) * n + j] += sx1 * pCorr;
                    v[i * n + j] -= sy0 * pCorr;
                    v[i * n + j + 1] += sy1 * pCorr;
                }
            }
        }
    }

    public void extrapolate() {
        int n = numY;
        for (int i = 0; i < numX; i++) {
            u[i * n + 0] = u[i * n + 1];
            u[i * n + numY - 1] = u[i * n + numY - 2];
        }
        for (int j = 0; j < numY; j++) {
            v[j] = v[n + j];                             // v[0*n+j] = v[1*n+j]
            v[(numX - 1) * n + j] = v[(numX - 2) * n + j];
        }
    }

    /** Bilinear sample of {@code field} (U_FIELD/V_FIELD/S_FIELD) at physical position (x,y). */
    public double sampleField(double x, double y, int field) {
        int n = numY;
        double h1 = 1.0 / h;
        double h2 = 0.5 * h;

        x = Math.max(Math.min(x, numX * h), h);
        y = Math.max(Math.min(y, numY * h), h);

        double dx = 0.0, dy = 0.0;
        double[] f;

        switch (field) {
            case U_FIELD -> { f = u; dy = h2; }
            case V_FIELD -> { f = v; dx = h2; }
            case S_FIELD -> { f = m; dx = h2; dy = h2; }
            default -> throw new IllegalArgumentException("Unknown field " + field);
        }

        int x0 = (int) Math.min(Math.floor((x - dx) * h1), numX - 1);
        double tx = ((x - dx) - x0 * h) * h1;
        int x1 = Math.min(x0 + 1, numX - 1);

        int y0 = (int) Math.min(Math.floor((y - dy) * h1), numY - 1);
        double ty = ((y - dy) - y0 * h) * h1;
        int y1 = Math.min(y0 + 1, numY - 1);

        double sx = 1.0 - tx;
        double sy = 1.0 - ty;

        return sx * sy * f[x0 * n + y0]
                + tx * sy * f[x1 * n + y0]
                + tx * ty * f[x1 * n + y1]
                + sx * ty * f[x0 * n + y1];
    }

    private double avgU(int i, int j) {
        int n = numY;
        return (u[i * n + j - 1] + u[i * n + j] + u[(i + 1) * n + j - 1] + u[(i + 1) * n + j]) * 0.25;
    }

    private double avgV(int i, int j) {
        int n = numY;
        return (v[(i - 1) * n + j] + v[i * n + j] + v[(i - 1) * n + j + 1] + v[i * n + j + 1]) * 0.25;
    }

    public void advectVel(double dt) {
        System.arraycopy(u, 0, newU, 0, u.length);
        System.arraycopy(v, 0, newV, 0, v.length);

        int n = numY;
        double h2 = 0.5 * h;

        for (int i = 1; i < numX; i++) {
            for (int j = 1; j < numY; j++) {

                // u component
                if (s[i * n + j] != 0.0 && s[(i - 1) * n + j] != 0.0 && j < numY - 1) {
                    double x = i * h;
                    double y = j * h + h2;
                    double uu = u[i * n + j];
                    double vv = avgV(i, j);
                    x -= dt * uu;
                    y -= dt * vv;
                    newU[i * n + j] = sampleField(x, y, U_FIELD);
                }
                // v component
                if (s[i * n + j] != 0.0 && s[i * n + j - 1] != 0.0 && i < numX - 1) {
                    double x = i * h + h2;
                    double y = j * h;
                    double uu = avgU(i, j);
                    double vv = v[i * n + j];
                    x -= dt * uu;
                    y -= dt * vv;
                    newV[i * n + j] = sampleField(x, y, V_FIELD);
                }
            }
        }

        System.arraycopy(newU, 0, u, 0, u.length);
        System.arraycopy(newV, 0, v, 0, v.length);
    }

    public void advectSmoke(double dt) {
        System.arraycopy(m, 0, newM, 0, m.length);

        int n = numY;
        double h2 = 0.5 * h;

        for (int i = 1; i < numX - 1; i++) {
            for (int j = 1; j < numY - 1; j++) {
                if (s[i * n + j] != 0.0) {
                    double uu = (u[i * n + j] + u[(i + 1) * n + j]) * 0.5;
                    double vv = (v[i * n + j] + v[i * n + j + 1]) * 0.5;
                    double x = i * h + h2 - dt * uu;
                    double y = j * h + h2 - dt * vv;
                    newM[i * n + j] = sampleField(x, y, S_FIELD);
                }
            }
        }
        System.arraycopy(newM, 0, m, 0, m.length);
    }

    public void simulate(double dt, double gravity, int numIters, double overRelaxation) {
        integrate(dt, gravity);
        Arrays.fill(p, 0.0);
        solveIncompressibility(numIters, dt, overRelaxation);
        extrapolate();
        advectVel(dt);
        advectSmoke(dt);
    }

    /** Max |divergence| over fluid cells, using the same stencil {@link #solveIncompressibility} zeroes -- diagnostic only. */
    public double maxDivergence() {
        int n = numY;
        double maxDiv = 0.0;
        for (int i = 1; i < numX - 1; i++) {
            for (int j = 1; j < numY - 1; j++) {
                if (s[i * n + j] == 0.0) continue;
                double div = u[(i + 1) * n + j] - u[i * n + j] + v[i * n + j + 1] - v[i * n + j];
                maxDiv = Math.max(maxDiv, Math.abs(div));
            }
        }
        return maxDiv;
    }
}