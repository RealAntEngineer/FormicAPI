package com.rae.formicapi.math_tests.solvers;

import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR2Tensor;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import com.rae.formicapi.foundation.math.solvers.NewtonKrylov;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DNSTest {

    @Test
    void laminarPoiseuilleFlow() {

        int nx = 32;
        int ny = 32;

        double viscosity = 0.1;
        double force = 1.0;

        int cells = nx * ny;

        PaddedCSR2Tensor C = new PaddedCSR2Tensor(cells * 2, 9);

        PaddedCSRMatrix A = new PaddedCSRMatrix(cells * 2, cells * 2, 5);

        double[] x = new double[cells * 2];
        double[] b = new double[cells * 2];

        /*
         * Build diffusion + boundary operator
         *
         * Unknown layout:
         *
         * x[2*cell]     = u
         * x[2*cell+1]   = v
         */
        for (int y = 1; y < ny - 1; y++) {

            for (int xx = 0; xx < nx; xx++) {

                int cell = y * nx + xx;

                int u = 2 * cell;
                int v = u + 1;

                 // Diffusion stencil ν ∇²u
                int left  = 2 * (y * nx + ((xx - 1 + nx) % nx));
                int right = 2 * (y * nx + ((xx + 1) % nx));

                int down = 2 * ((y - 1) * nx + xx);
                int up   = 2 * ((y + 1) * nx + xx);

                A.setRow(u,
                        new double[]{-4 * viscosity, viscosity, viscosity, viscosity, viscosity},
                        new int[]{u, left, right, down, up}, 5);

                A.setRow(v,
                        new double[]{-4 * viscosity, viscosity, viscosity, viscosity, viscosity},
                        new int[]{v, left + 1, right + 1, down + 1, up + 1}, 5);
            }
        }

        /*
         * Body force in x direction
         */
        for (int y = 1; y < ny - 1; y++) {
            for (int xx = 0; xx < nx; xx++) {

                int cell = y * nx + xx;

                b[2 * cell] = -force;
                b[2 * cell + 1] = 0;
            }
        }

        /*
         * Non-slip walls
         */
        for (int xx = 0; xx < nx; xx++) {

            int bottom = xx;
            int top = (ny - 1) * nx + xx;

            x[2 * bottom] = 0;
            x[2 * bottom + 1] = 0;
            x[2 * top] = 0;
            x[2 * top + 1] = 0;
        }

        double[] Ax = new double[cells * 2];
        double[] F = new double[cells * 2];
        double[] dx = new double[cells * 2];

        double[] r = new double[cells * 2];
        double[] rHat = new double[cells * 2];
        double[] p = new double[cells * 2];
        double[] v = new double[cells * 2];
        double[] s = new double[cells * 2];
        double[] t = new double[cells * 2];


        NewtonKrylov.solve(C, A, x, b,
                20, 200, 1e-8, 1e-10,
                Ax, F, dx, r, rHat, p, v, s, t);



        //print it here

        double[][] velocity = new double[nx][ny];

        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                velocity[i][j] = x[2 * (j * nx + i)];//u

        Field2DRenderer.saveHeatmap(velocity, "laminar_2d_velocity.png");
        /*
         * Check quadratic velocity profile
         */

        double H = ny - 1;

        for (int y = 1; y < ny - 1; y++) {

            double expected = force / (2 * viscosity) * y * (H - y);
            double actual = x[2 * (y * nx)];

            assertEquals(expected, actual, expected * 0.05);
        }

        /*
         * No transverse velocity
         */
        for (int i = 0; i < cells; i++)
            assertEquals(0, x[2 * i + 1], 1e-8);
    }

    // =====================================================================
    // Implicit (backward Euler) time integration.
    //
    //   (x^{n+1} - x^n)/dt + C:x^{n+1}:x^{n+1} + A*x^{n+1} - b = 0
    //
    // is linear in the unsteady term, so it's folded directly into the
    // existing steady operator:
    //
    //   A_eff = A + M/dt        (M = identity on u,v dofs, 0 on p dofs)
    //   b_eff = b + M*x^n/dt
    //
    // and solved each step with the *same* NewtonKrylov.solve used for the
    // steady case. Because dt and viscosity are fixed, A_eff is built once;
    // only b_eff (and the warm-start x) change per step. Backward Euler is
    // unconditionally linearly stable, but Newton's convergence radius for
    // an advection-dominated step still degrades if dt is too large -- if
    // Newton starts needing many more iterations or failing to converge,
    // shrink dt rather than raising maxNewtonIter.
    // =====================================================================

    @Test
    void lidDrivenCavityTurbulentDNSTransient() {

        int nx = 64;
        int ny = 64;

        double viscosity = 0.1;// -> make it so it's not possible to have
        double pressurePenalty = 1e-2;//This break the simulation too easily

        double dt = 1d/50;
        int nSteps = 5000;
        int printEvery = 100;

        int cells = nx * ny;
        int n = cells * 3;

        PaddedCSR2Tensor C = new PaddedCSR2Tensor(n, 4);
        PaddedCSRMatrix A = new PaddedCSRMatrix(n, n, 7);

        double[] x = new double[n];
        double[] b = new double[n];

        boolean[] isBoundary = new boolean[n];   // dofs with a Dirichlet row (u,v walls) or Neumann pressure row
        boolean[] isPressure = new boolean[n];   // p dofs (no mass term, b term always 0)
        double[] bcValue = new double[n];        // fixed value for Dirichlet u,v boundary dofs


        double[] Ax = new double[n];
        double[] F = new double[n];

        double[] dx = new double[n];
        double[] r = new double[n];
        double[] rHatO = new double[n];
        double[] vBuf = new double[n];
        double[] pBuf = new double[n];
        double[] sBuf = new double[n];
        double[] tBuf = new double[n];

        // Build the steady operator with the mass/dt term baked into the
        // diagonal of the interior u,v rows.
        buildNavierStokesOperator(nx, ny, viscosity, pressurePenalty, 1.0 / dt, C, A);

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {

                int cell = j * nx + i;
                int u = 3 * cell, v = u + 1, p = u + 2;

                isPressure[p] = true;

                boolean onBoundary = (i == 0 || i == nx - 1 || j == 0 || j == ny - 1);

                if (onBoundary) {
                    isBoundary[u] = true;
                    isBoundary[v] = true;

                    double uWall = 0, vWall = 0;
                    if (j == ny - 1) uWall = 1.0; // lid moves right

                    bcValue[u] = uWall;
                    bcValue[v] = vWall;

                    x[u] = uWall;
                    x[v] = vWall;
                } else {
                    x[u] = 0.1;
                    x[v] = 0.1;
                }
            }
        }


        // -----------------------------------------------------------
        // XChart setup: one window for convergence (log-scale residual
        // + Newton iteration count), one for flow diagnostics (max
        // velocity, TKE proxy). Kept separate because the residual
        // spans many orders of magnitude and would flatten the
        // velocity/TKE curves to invisibility on a shared linear axis.
        // -----------------------------------------------------------

        /*XYChart convergenceChart = new XYChartBuilder()
                .width(900).height(500)
                .title("Newton convergence per time step")
                .xAxisTitle("t").yAxisTitle("final residual (log)")
                .build();
        convergenceChart.getStyler().setYAxisLogarithmic(true);
        convergenceChart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);
        convergenceChart.addSeries("final residual", new double[]{0}, new double[]{1});
        convergenceChart.addSeries("newton iterations", new double[]{0}, new double[]{1});
        //        .setYAxisGroup(1);
        //convergenceChart.setYAxisGroupTitle(1, "newton iterations");
        //convergenceChart.getStyler().gr(1, Styler.TextAlignment.Right);

        XYChart diagnosticsChart = new XYChartBuilder()
                .width(900).height(500)
                .title("Flow diagnostics")
                .xAxisTitle("t").yAxisTitle("value")
                .build();
        diagnosticsChart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);
        diagnosticsChart.addSeries("max |velocity|", new double[]{0}, new double[]{0});
        diagnosticsChart.addSeries("TKE proxy", new double[]{0}, new double[]{0});

        SwingWrapper<XYChart> convergenceWindow = new SwingWrapper<>(convergenceChart).displayChart();
        SwingWrapper<XYChart> diagnosticsWindow = new SwingWrapper<>(diagnosticsChart).displayChart();

        List<Double> tAll          = new ArrayList<>();
        List<Double> residualAll   = new ArrayList<>();
        List<Double> newtonIterAll = new ArrayList<>();

        List<Double> tSampled = new ArrayList<>();
        List<Double> maxSpeedSampled = new ArrayList<>();
        List<Double> tkeSampled = new ArrayList<>();*/


        NewtonKrylov.Stats stats = new NewtonKrylov.Stats();
        long startTime;
        for (int step = 0; step < nSteps; step++) {
            startTime = System.nanoTime();

            // Assemble b_eff for this step: boundary dofs hold their fixed
            // BC value, pressure dofs are always 0 (no time derivative),
            // interior u/v dofs get x^n / dt.
            for (int i = 0; i < n; i++) {
                if (isBoundary[i]) b[i] = bcValue[i];
                else if (isPressure[i]) b[i] = 0;
                else b[i] = x[i] / dt;//?? what is this
            }

            stats.reset();


            // x already holds x^n and is reused as the warm start for x^{n+1}-
            NewtonKrylov.solve(C, A, x, b, 20, 200, 1e-5 * nx * ny, 1e-6 * nx * ny,
            Ax, F, dx, r, rHatO, vBuf, pBuf, sBuf, tBuf, stats);

            if (step % printEvery == 0){
                long newTime = System.nanoTime();
                System.out.printf("Newton converged=%b in %d iters%n, it took %3.3e ns per cells%n",
                        stats.converged, stats.newtonIterationCount, ((float) (newTime - startTime)/ cells));

                for (int i = 0; i < stats.newtonResiduals.size(); i++)
                    System.out.printf("  iter %d: residual=%.3e, linear iters=%d%n",
                            i, stats.newtonResiduals.get(i), (stats.linearIterations.size() - 1 >= i ? stats.linearIterations.get(i) : 0));
            }



            if (step % printEvery == 0 || step == nSteps - 1) {

                double[][] u = new double[nx][ny];
                double[][] v = new double[nx][ny];
                double[][] speed = new double[nx][ny];

                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++) {
                        int c = j * nx + i;
                        u[i][j] = x[3 * c];
                        v[i][j] = x[3 * c + 1];
                        speed[i][j] = Math.hypot(u[i][j], v[i][j]);
                    }

                //System.out.printf("t = %.4f (step %d/%d)%n", step * dt, step, nSteps);
                //printDiagnostics(nx, ny, u, v, speed);

                Field2DRenderer.saveHeatmap(speed, String.format("cavity/speed_%04d.png", step), Field2DRenderer.Interpolation.NEAREST);
            }
        }
    }

    /**
     * Prints velocity-amplitude and turbulence-proxy statistics for the
     * converged field.
     *
     * NOTE: because this is a single steady-state solution (not a time
     * series or ensemble), "turbulence" here means spatial fluctuation
     * about the domain-mean velocity -- a proxy (spatial TKE / enstrophy),
     * not a real turbulence statistic. True turbulence intensity requires
     * averaging over time (or many realizations), which needs unsteady
     * time-stepping through NewtonKrylov rather than one static solve.
     */
    static void printDiagnostics(int nx, int ny, double[][] u, double[][] v, double[][] speed) {

        int cells = nx * ny;

        double meanU = 0, meanV = 0, maxSpeed = 0;

        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++) {
                meanU += u[i][j];
                meanV += v[i][j];
                maxSpeed = Math.max(maxSpeed, speed[i][j]);
            }

        meanU /= cells;
        meanV /= cells;

        double tke = 0;          // spatial proxy for turbulent kinetic energy
        double enstrophy = 0;    // 0.5 * vorticity^2, integrated -- a common turbulence-content diagnostic

        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {

                double up = u[i][j] - meanU;
                double vp = v[i][j] - meanV;
                tke += 0.5 * (up * up + vp * vp);

                double dvdx = (v[i + 1][j] - v[i - 1][j]) * 0.5;
                double dudy = (u[i][j + 1] - u[i][j - 1]) * 0.5;
                double vorticity = dvdx - dudy;

                enstrophy += 0.5 * vorticity * vorticity;
            }
        }

        tke /= cells;
        enstrophy /= cells;

        System.out.printf("Max |velocity|         : %.6f%n", maxSpeed);
        System.out.printf("Mean velocity (u,v)    : (%.6f, %.6f)%n", meanU, meanV);
        System.out.printf("Spatial TKE proxy      : %.6e%n", tke);
        System.out.printf("Enstrophy (turb. proxy): %.6e%n", enstrophy);
    }

    static void buildNavierStokesOperator(int nx, int ny, double nu, double pressurePenalty,
                                          double invDt, PaddedCSR2Tensor C, PaddedCSRMatrix A) {
        // invDt = 1/dt for implicit time stepping (added to the u,v diagonal
        // as a mass term), or 0 for a pure steady-state solve.

        double[] values = new double[4];
        int[] var1 = new int[4];
        int[] var2 = new int[4];

        double[] laplace = new double[7];
        int[] cols = new int[7];

        double[] pRow = new double[5];
        int[] pCols = new int[5];

        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {

                int cell = y * nx + x;

                int u = 3 * cell;
                int v = u + 1;
                int p = u + 2;

                // -------------------------------------------------
                // Boundary equations
                // -------------------------------------------------

                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {

                    A.setRow(u, new double[]{1}, new int[]{u}, 1);
                    A.setRow(v, new double[]{1}, new int[]{v}, 1);

                    C.setRow(u, new double[0], new int[0], new int[0], 0);// what ? a 0 count
                    C.setRow(v, new double[0], new int[0], new int[0], 0);
                    C.setRow(p, new double[0], new int[0], new int[0], 0);

                    // Zero-normal-gradient (Neumann) pressure BC: copy from the
                    // nearest interior-facing neighbor. Approximate at corners
                    // (picks the x-direction neighbor first).
                    int pNeighbor;
                    if (x == 0) pNeighbor = p + 3;
                    else if (x == nx - 1) pNeighbor = p - 3;
                    else if (y == 0) pNeighbor = p + 3 * nx;
                    else pNeighbor = p - 3 * nx;

                    A.setRow(p, new double[]{1, -1}, new int[]{p, pNeighbor}, 2);

                    continue;
                }

                int west = 3 * (cell - 1);
                int east = 3 * (cell + 1);
                int south = 3 * (cell - nx);
                int north = 3 * (cell + nx);

                // -------------------------------------------------
                // u-momentum: viscous Laplacian + pressure gradient
                // -------------------------------------------------

                laplace[0] = 4 * nu + invDt;
                laplace[1] = -nu;
                laplace[2] = -nu;
                laplace[3] = -nu;
                laplace[4] = -nu;
                laplace[5] = 0.5;   // +dp/dx : p at east
                laplace[6] = -0.5;  // -dp/dx : p at west

                cols[0] = u;
                cols[1] = west;
                cols[2] = east;
                cols[3] = south;
                cols[4] = north;
                cols[5] = east + 2;
                cols[6] = west + 2;

                A.setRow(u, laplace, cols, 7);

                // -------------------------------------------------
                // v-momentum: viscous Laplacian + pressure gradient
                // -------------------------------------------------

                laplace[5] = 0.5;   // +dp/dy : p at north
                laplace[6] = -0.5;  // -dp/dy : p at south

                cols[0] = v;
                cols[1] = west + 1;
                cols[2] = east + 1;
                cols[3] = south + 1;
                cols[4] = north + 1;
                cols[5] = north + 2;
                cols[6] = south + 2;

                A.setRow(v, laplace, cols, 7);

                // -------------------------------------------------
                // continuity (pressure) equation:
                //   du/dx + dv/dy + eps*p = 0
                // -------------------------------------------------

                pRow[0] = 0.5;
                pRow[1] = -0.5;
                pRow[2] = 0.5;
                pRow[3] = -0.5;
                pRow[4] = pressurePenalty;

                pCols[0] = east;       // u east
                pCols[1] = west;       // u west
                pCols[2] = north + 1;  // v north
                pCols[3] = south + 1;  // v south
                pCols[4] = p;

                A.setRow(p, pRow, pCols, 5);

                // -------------------------------------------------
                // u-equation advection: u du/dx + v du/dy   (unchanged, quadratic -> C)
                // -------------------------------------------------

                values[0] = 0.5;
                values[1] = -0.5;
                values[2] = 0.5;
                values[3] = -0.5;

                var1[0] = u;
                var2[0] = east;

                var1[1] = u;
                var2[1] = west;

                var1[2] = v;
                var2[2] = north;

                var1[3] = v;
                var2[3] = south;

                C.setRow(u, values, var1, var2, 4);

                // -------------------------------------------------
                // v-equation advection: u dv/dx + v dv/dy
                // -------------------------------------------------

                var1[0] = u;
                var2[0] = east + 1;

                var1[1] = u;
                var2[1] = west + 1;

                var1[2] = v;
                var2[2] = north + 1;

                var1[3] = v;
                var2[3] = south + 1;

                C.setRow(v, values, var1, var2, 4);

                //No non-linearity on P, so we miss a bit of sparsity on C
            }
        }
    }
}
