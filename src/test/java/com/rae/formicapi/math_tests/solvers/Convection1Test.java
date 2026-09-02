package com.rae.formicapi.math_tests.solvers;

import com.rae.formicapi.LiveChartWindow;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR3Tensor;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;
import com.rae.formicapi.foundation.math.solvers.NewtonKrylov;
import org.junit.jupiter.api.Test;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.Styler;

import java.util.ArrayList;
import java.util.List;

public class Convection1Test {

    // =====================================================================
    // Natural convection on a horizontal plate (Rayleigh-Bénard convection).
    //
    // Unknown layout: 4 dof / cell -- u = 4*cell, v = 4*cell+1, p = 4*cell+2,
    // T = 4*cell+3.
    //
    // Domain: unit square. Bottom plate (y=0) held at T_hot, top (y=ny-1)
    // held at T_cold. Side walls (x=0, x=nx-1) thermally insulated
    // (zero-gradient). All four walls no-slip (u=v=0). Pressure Neumann
    // on all walls, as before.
    //
    // Governing equations (Boussinesq):
    //   du/dt + u*du/dx + v*du/dy - nu*Lap(u) + dp/dx               = 0
    //   dv/dt + u*dv/dx + v*dv/dy - nu*Lap(v) + dp/dy - g*beta*(T-T0) = 0
    //   div(u) + eps*p                                               = 0
    //   dT/dt + u*dT/dx + v*dT/dy - alpha*Lap(T)                     = 0
    //
    // Buoyancy is linear in T, so it's one extra entry in A's v-row
    // (coefficient -g*beta at the T dof) plus a constant -g*beta*T0 folded
    // into b each step alongside the mass/dt term. Advection of T is
    // quadratic (u*T, v*T products) so it goes in C, structurally
    // identical to the momentum advection terms.
    // =====================================================================

    enum DofRole {
        INTERIOR_U, INTERIOR_V, INTERIOR_P, INTERIOR_T,
        DIRICHLET,   // fixed value in bcValue (u/v walls = 0, T plates = Thot/Tcold)
        NEUMANN      // zero-gradient row already baked into A; b is always 0
    }

    static void buildBoussinesqOperator(int nx, int ny, double nu, double alpha, double beta, double g,
                                        double tHot, double tCold, double pressurePenalty, double invDt,
                                        PaddedCSR3Tensor C, PaddedCSRMatrix A, DofRole[] role, double[] bcValue) {

        double hx = 1.0 / (nx - 1);
        double hy = 1.0 / (ny - 1);

        double invHx = 1.0 / hx;
        double invHy = 1.0 / hy;
        double invHx2 = invHx * invHx;
        double invHy2 = invHy * invHy;

        double buoyancyCoeff = g * beta;

        double[] values = new double[4];
        int[] var1 = new int[4];
        int[] var2 = new int[4];

        double[] uRow = new double[7];
        int[] uCols = new int[7];

        double[] vRow = new double[8];
        int[] vCols = new int[8];

        double[] pRow = new double[5];
        int[] pCols = new int[5];

        double[] tRow = new double[5];
        int[] tCols = new int[5];

        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {

                int cell = y * nx + x;
                int u = 4 * cell, v = u + 1, p = u + 2, T = u + 3;

                boolean isPlate = (y == 0 || y == ny - 1);
                boolean isSide = (x == 0 || x == nx - 1);

                // ---------------- velocity: no-slip on all 4 walls ----------------
                if (isPlate || isSide) {
                    A.setRow(u, new double[]{1}, new int[]{u}, 1);
                    A.setRow(v, new double[]{1}, new int[]{v}, 1);
                    C.setRow(u, new double[0], new int[0], new int[0], 0);
                    C.setRow(v, new double[0], new int[0], new int[0], 0);
                    role[u] = DofRole.DIRICHLET;
                    role[v] = DofRole.DIRICHLET;
                    bcValue[u] = 0;
                    bcValue[v] = 0;
                } else {
                    role[u] = DofRole.INTERIOR_U;
                    role[v] = DofRole.INTERIOR_V;
                }

                // ---------------- pressure: Neumann on all 4 walls ----------------
                if (isPlate || isSide) {
                    int pNeighbor;
                    if (x == 0) pNeighbor = p + 4;
                    else if (x == nx - 1) pNeighbor = p - 4;
                    else if (y == 0) pNeighbor = p + 4 * nx;
                    else pNeighbor = p - 4 * nx;

                    A.setRow(p, new double[]{1, -1}, new int[]{p, pNeighbor}, 2);
                    C.setRow(p, new double[0], new int[0], new int[0], 0);
                    role[p] = DofRole.NEUMANN;
                } else {
                    role[p] = DofRole.INTERIOR_P;
                }

                // ---------------- temperature: Dirichlet plates, Neumann sides ----------------
                if (isPlate) {
                    double tVal = (y == 0) ? tHot : tCold;
                    A.setRow(T, new double[]{1}, new int[]{T}, 1);
                    C.setRow(T, new double[0], new int[0], new int[0], 0);
                    role[T] = DofRole.DIRICHLET;
                    bcValue[T] = tVal;
                } else if (isSide) {
                    int tNeighbor = (x == 0) ? T + 4 : T - 4;
                    A.setRow(T, new double[]{1, -1}, new int[]{T, tNeighbor}, 2);
                    C.setRow(T, new double[0], new int[0], new int[0], 0);
                    role[T] = DofRole.NEUMANN;
                } else {
                    role[T] = DofRole.INTERIOR_T;
                }

                if (isPlate || isSide)
                    continue;

                // ---------------- interior cell: full coupled physics ----------------

                int west = 4 * (cell - 1), east = 4 * (cell + 1);
                int south = 4 * (cell - nx), north = 4 * (cell + nx);

                // u-momentum: viscous Laplacian + pressure gradient
                uRow[0] = 2 * nu * (invHx2 + invHy2) + invDt;
                uRow[1] = -nu * invHx2;
                uRow[2] = -nu * invHx2;
                uRow[3] = -nu * invHy2;
                uRow[4] = -nu * invHy2;
                uRow[5] = 0.5 * invHx;
                uRow[6] = -0.5 * invHx;

                uCols[0] = u;
                uCols[1] = west;
                uCols[2] = east;
                uCols[3] = south;
                uCols[4] = north;
                uCols[5] = east + 2;
                uCols[6] = west + 2;

                A.setRow(u, uRow, uCols, 7);

                // v-momentum: viscous Laplacian + pressure gradient + buoyancy
                vRow[0] = 2 * nu * (invHx2 + invHy2) + invDt;
                vRow[1] = -nu * invHx2;
                vRow[2] = -nu * invHx2;
                vRow[3] = -nu * invHy2;
                vRow[4] = -nu * invHy2;
                vRow[5] = 0.5 * invHy;
                vRow[6] = -0.5 * invHy;
                vRow[7] = -buoyancyCoeff;

                vCols[0] = v;
                vCols[1] = west + 1;
                vCols[2] = east + 1;
                vCols[3] = south + 1;
                vCols[4] = north + 1;
                vCols[5] = north + 2;
                vCols[6] = south + 2;
                vCols[7] = T;

                A.setRow(v, vRow, vCols, 8);

                // continuity: div(u) + eps*p
                pRow[0] = 0.5 * invHx;
                pRow[1] = -0.5 * invHx;
                pRow[2] = 0.5 * invHy;
                pRow[3] = -0.5 * invHy;
                pRow[4] = pressurePenalty;

                pCols[0] = east;
                pCols[1] = west;
                pCols[2] = north + 1;
                pCols[3] = south + 1;
                pCols[4] = p;

                A.setRow(p, pRow, pCols, 5);

                // energy: thermal diffusion
                tRow[0] = 2 * alpha * (invHx2 + invHy2) + invDt;
                tRow[1] = -alpha * invHx2;
                tRow[2] = -alpha * invHx2;
                tRow[3] = -alpha * invHy2;
                tRow[4] = -alpha * invHy2;

                tCols[0] = T;
                tCols[1] = west + 3;
                tCols[2] = east + 3;
                tCols[3] = south + 3;
                tCols[4] = north + 3;

                A.setRow(T, tRow, tCols, 5);

                // advection: u-momentum, v-momentum, energy (quadratic -> C)
                values[0] = 0.5 * invHx;
                values[1] = -0.5 * invHx;
                values[2] = 0.5 * invHy;
                values[3] = -0.5 * invHy;

                var1[0] = u; var2[0] = east;
                var1[1] = u; var2[1] = west;
                var1[2] = v; var2[2] = north;
                var1[3] = v; var2[3] = south;
                C.setRow(u, values, var1, var2, 4);

                var1[0] = u; var2[0] = east + 1;
                var1[1] = u; var2[1] = west + 1;
                var1[2] = v; var2[2] = north + 1;
                var1[3] = v; var2[3] = south + 1;
                C.setRow(v, values, var1, var2, 4);

                var1[0] = u; var2[0] = east + 3;
                var1[1] = u; var2[1] = west + 3;
                var1[2] = v; var2[2] = north + 3;
                var1[3] = v; var2[3] = south + 3;
                C.setRow(T, values, var1, var2, 4);
            }
        }
    }

    //@Test
    void naturalConvectionHorizontalPlate() {

        int nx = 64;
        int ny = 64;

        double nu = 0.01;      // kinematic viscosity
        double alpha = 0.01;   // thermal diffusivity (Pr = nu/alpha = 1)
        double beta = 1.0;     // thermal expansion coefficient
        double g = 1.0;        // gravity (idealized, unit-square domain)

        double tHot = 1.0;     // bottom plate
        double tCold = 0.5;    // top plate
        double t0 = 0.5 * (tHot + tCold);
        double deltaT = tHot - tCold;

        double pressurePenalty = 1e-1;

        double dt = 1d/20d;
        int nSteps = 200;
        int printEvery = 50;
        int sampleEvery = 5;

        // Rayleigh number, L=1 (unit square domain): governs whether
        // convection sets in at all. Critical Ra for rigid-rigid boundaries
        // is ~1708 -- comfortably below the value below, so this setup
        // should develop convection rolls rather than stay conductive.
        double rayleigh = g * beta * deltaT / (nu * alpha);
        System.out.printf("Rayleigh number: %.1f (critical ~1708)%n", rayleigh);

        int cells = nx * ny;
        int n = cells * 4;

        PaddedCSR3Tensor C = new PaddedCSR3Tensor(n, 4);
        PaddedCSRMatrix  A = new PaddedCSRMatrix(n, n, 8);

        double[] x = new double[n];
        double[] b = new double[n];

        DofRole[] role = new DofRole[n];
        double[] bcValue = new double[n];

        buildBoussinesqOperator(nx, ny, nu, alpha, beta, g, tHot, tCold, pressurePenalty, 1.0 / dt, C, A, role, bcValue);

        // Initial condition: fluid at rest, linear conductive temperature
        // stratification top-to-bottom, plus a small random perturbation to
        // break symmetry -- a perfectly symmetric initial state has no way
        // to spontaneously form convection rolls under exact arithmetic.
        java.util.Random rnd = new java.util.Random(42);

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {

                int cell = j * nx + i;
                int u = 4 * cell, v = u + 1, T = u + 3;

                double frac = (double) j / (ny - 1); // 0 at bottom, 1 at top
                double tLinear = tHot + frac * (tCold - tHot);

                if (role[T] == DofRole.INTERIOR_T)
                    x[T] = tLinear + (rnd.nextDouble() - 0.5) * 0.01 * deltaT;
                else if (role[T] == DofRole.DIRICHLET)
                    x[T] = bcValue[T];

                if (role[u] == DofRole.DIRICHLET) x[u] = bcValue[u];
                if (role[v] == DofRole.DIRICHLET) x[v] = bcValue[v];
            }
        }

        double buoyancyConst = g * beta * t0; // -g*beta*T0 term folded into b for interior v-rows

        // -----------------------------------------------------------
        // XChart: separate single-axis charts (no dual-axis Styler calls,
        // which vary across XChart versions) -- Nusselt number and max
        // velocity tracked over time.
        // -----------------------------------------------------------

        XYChart nusseltChart = new XYChartBuilder()
                .width(900).height(400)
                .title("Nusselt number (Ra=" + (long) rayleigh + ")")
                .xAxisTitle("t").yAxisTitle("Nu")
                .build();
        nusseltChart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);
        nusseltChart.addSeries("Nusselt number", new double[]{0}, new double[]{1});

        XYChart velocityChart = new XYChartBuilder()
                .width(900).height(400)
                .title("Max velocity")
                .xAxisTitle("t").yAxisTitle("max |velocity|")
                .build();
        velocityChart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);
        velocityChart.addSeries("max |velocity|", new double[]{0}, new double[]{0});

        LiveChartWindow nusseltWindow = new LiveChartWindow(nusseltChart);
        LiveChartWindow velocityWindow = new LiveChartWindow(velocityChart);

        List<Double> tSampled       = new ArrayList<>();
        List<Double> nusseltSampled = new ArrayList<>();
        List<Double> maxSpeedSampled = new ArrayList<>();

        NewtonKrylov.Stats stats = new NewtonKrylov.Stats();

        for (int step = 0; step < nSteps; step++) {

            for (int i = 0; i < n; i++) {
                switch (role[i]) {
                    case DIRICHLET -> b[i] = bcValue[i];
                    case NEUMANN, INTERIOR_P -> b[i] = 0;
                    case INTERIOR_U, INTERIOR_T -> b[i] = x[i] / dt;
                    case INTERIOR_V -> b[i] = x[i] / dt - buoyancyConst;
                }
            }

            stats.reset();

            NewtonKrylov.solve(C, A, new CpuDoubleVector(x), new CpuDoubleVector(b), 20, 200,
                    1e-4 * Math.sqrt((double) nx * ny), 1e-5 * Math.sqrt((double) nx * ny), null, stats, null,
                    new WorkingBuffer<>(7, () -> new CpuDoubleVector(n)),
                    new WorkingBuffer<>(3, () -> new CpuDoubleVector(n)));

            double t = step * dt;

            if (step % sampleEvery == 0 || step == nSteps - 1) {

                double maxSpeed = 0;
                double convectiveFlux = 0; // <v*T> averaged over interior cells

                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++) {

                        int cell = j * nx + i;
                        int u = 4 * cell, v = u + 1, T = u + 3;

                        maxSpeed = Math.max(maxSpeed, Math.hypot(x[u], x[v]));

                        if (role[T] == DofRole.INTERIOR_T)
                            convectiveFlux += x[v] * (x[T] - t0);
                    }

                convectiveFlux /= cells;

                // Nu = 1 (pure conduction) + convective enhancement.
                // Standard nondimensional form: Nu = 1 + <v*T'>*L / (alpha*deltaT)
                double nusselt = 1.0 + convectiveFlux / (alpha * deltaT);

                tSampled.add(t);
                nusseltSampled.add(nusselt);
                maxSpeedSampled.add(maxSpeed);
            }

            if (step % printEvery == 0 || step == nSteps - 1) {

                nusseltChart.updateXYSeries("Nusselt number", toArray(tSampled), toArray(nusseltSampled), null);
                nusseltWindow.repaint();

                velocityChart.updateXYSeries("max |velocity|", toArray(tSampled), toArray(maxSpeedSampled), null);
                velocityWindow.repaint();

                // Headless (e.g. CI): no live window, so drop periodic PNG
                // checkpoints instead of silently producing nothing to check.
                if (!nusseltWindow.isInteractive()) {
                    nusseltWindow.saveSnapshot("nusselt_latest.png");
                    velocityWindow.saveSnapshot("velocity_latest.png");
                }

                System.out.printf("t=%.2f step=%d/%d newtonIters=%d converged=%b Nu=%.4f%n",
                        t, step, nSteps, stats.newtonIterationCount, stats.converged,
                        nusseltSampled.isEmpty() ? Double.NaN : nusseltSampled.get(nusseltSampled.size() - 1));

                double[][] speed = new double[nx][ny];
                double[][] temperature = new double[nx][ny];

                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++) {
                        int cell = j * nx + i;
                        int u = 4 * cell, v = u + 1, T = u + 3;
                        speed[i][j] = Math.hypot(x[u], x[v]);
                        temperature[i][j] = x[T];
                    }

                Field2DRenderer.saveHeatmap(speed, String.format("convection_RB/speed_%05d.png", step),
                        Field2DRenderer.Interpolation.NEAREST);
                Field2DRenderer.saveHeatmap(temperature, String.format("convection_RB/temp_%05d.png", step),
                        Field2DRenderer.Interpolation.NEAREST);
            }
        }
    }

    private static double[] toArray(List<Double> list) {
        double[] arr = new double[list.size()];
        for (int i = 0; i < arr.length; i++)
            arr[i] = list.get(i);
        return arr;
    }
}
