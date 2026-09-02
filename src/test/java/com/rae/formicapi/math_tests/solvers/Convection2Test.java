package com.rae.formicapi.math_tests.solvers;

import com.rae.formicapi.LiveChartWindow;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR3Tensor;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;
import com.rae.formicapi.foundation.math.solvers.NewtonKrylov;
import org.junit.jupiter.api.Test;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.Styler;

import java.util.ArrayList;
import java.util.List;

import static com.google.common.primitives.Doubles.toArray;

public class Convection2Test {

    enum DofRole {
        INTERIOR_U, INTERIOR_V, INTERIOR_P, INTERIOR_T,
        DIRICHLET,   // fixed value in bcValue (u/v walls = 0, T plates = Thot/Tcold)
        NEUMANN      // zero-gradient row already baked into A; b is always 0
    }

    // =====================================================================
    // Natural convection from a single heated plate suspended mid-domain.
    //
    // Unlike Rayleigh-Bénard (heated layer, hot bottom / cold top over the
    // whole domain), this is an isolated heated strip acting as an internal
    // obstacle: solid (u=v=0, T=T_hot) on a horizontal band of cells in the
    // middle of the domain, surrounded by fluid, inside an enclosure whose
    // outer walls are no-slip and held at ambient temperature.
    //
    // Fluid cells touching the plate need no special handling: the existing
    // 5-point stencils already reference neighboring cells by grid index, so
    // a fluid cell next to the plate automatically picks up the plate's
    // pinned u=v=0, T=T_hot through its ordinary Laplacian/advection terms.
    // That's a simple immersed-boundary method, for free.
    //
    // No critical-Ra threshold applies to this geometry the way it does for
    // an enclosed heated layer -- an upward-facing heated plate in open
    // fluid is convectively unstable for any T_hot > T_ambient, so a plume
    // should start forming essentially immediately.
    // =====================================================================

    static void buildHeatedPlateOperator(int nx, int ny, double hx, double hy, double nu, double alpha, double beta, double g,
                                         double tHot, double tAmbient,
                                         int plateRow, int plateColStart, int plateColEnd,
                                         double pressurePenalty, double invDt,
                                         PaddedCSR3Tensor C, PaddedCSRMatrix A, double[] scaling, DofRole[] role, double[] bcValue) {

        //double hx = 1.0 / (nx - 1);
        //double hy = 1.0 / (ny - 1);

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

                boolean isOuterWall = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);
                boolean isPlate = !isOuterWall && (y == plateRow) && (x >= plateColStart) && (x <= plateColEnd);

                scaling[u] = 1;
                scaling[v] = 1;
                scaling[p] = 1/10.0;
                scaling[T] = 10;

                if (isOuterWall || isPlate) {

                    // Solid everywhere here: no-slip velocity.
                    A.setRow(u, new double[]{1}, new int[]{u}, 1);
                    A.setRow(v, new double[]{1}, new int[]{v}, 1);
                    C.setRow(u, new double[0], new int[0], new int[0], 0);
                    C.setRow(v, new double[0], new int[0], new int[0], 0);
                    role[u] = DofRole.DIRICHLET;
                    role[v] = DofRole.DIRICHLET;
                    bcValue[u] = 0;
                    bcValue[v] = 0;

                    // Temperature: ambient at the enclosure walls, hot at the plate.
                    A.setRow(T, new double[]{1}, new int[]{T}, 1);
                    C.setRow(T, new double[0], new int[0], new int[0], 0);
                    role[T] = DofRole.DIRICHLET;
                    bcValue[T] = isPlate ? tHot : tAmbient;

                    // Pressure: physically meaningless inside solid regions.
                    // Plate -> pin to 0 (decoupled; continuity never reads a
                    // neighbor's p, only its u/v, so this has no side effect
                    // on the surrounding fluid). Outer wall -> Neumann
                    // (zero-gradient) toward the nearest interior neighbor,
                    // as before.
                    if (isPlate) {
                        A.setRow(p, new double[]{1}, new int[]{p}, 1);
                        role[p] = DofRole.DIRICHLET;
                        bcValue[p] = 0;
                    } else {
                        int pNeighbor;
                        if (x == 0) pNeighbor = p + 4;
                        else if (x == nx - 1) pNeighbor = p - 4;
                        else if (y == 0) pNeighbor = p + 4 * nx;
                        else pNeighbor = p - 4 * nx;

                        A.setRow(p, new double[]{1, -1}, new int[]{p, pNeighbor}, 2);
                        role[p] = DofRole.NEUMANN;
                    }
                    C.setRow(p, new double[0], new int[0], new int[0], 0);

                    continue;
                }

                // ---------------- fluid cell: full coupled physics ----------------


                role[u] = DofRole.INTERIOR_U;
                role[v] = DofRole.INTERIOR_V;
                role[p] = DofRole.INTERIOR_P;
                role[T] = DofRole.INTERIOR_T;

                int west = 4 * (cell - 1), east = 4 * (cell + 1);
                int south = 4 * (cell - nx), north = 4 * (cell + nx);

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

                // div(u*u, u*v)
                values[0] =  0.5 * invHx;
                values[1] = -0.5 * invHx;
                values[2] =  0.5 * invHy;
                values[3] = -0.5 * invHy;

                // d(u²)/dx
                var1[0] = u;
                var2[0] = east;

                var1[1] = u;
                var2[1] = west;

                // d(uv)/dy
                var1[2] = u;
                var2[2] = north + 1;   // u*v north

                var1[3] = u;
                var2[3] = south + 1;   // u*v south

                C.setRow(u, values, var1, var2, 4);

                // d(uv)/dx
                var1[0] = u;
                var2[0] = east + 1;   // u_e*v_e

                var1[1] = u;
                var2[1] = west + 1;   // u_w*v_w

                // d(v²)/dy
                var1[2] = v;
                var2[2] = north + 1;  // v_n²

                var1[3] = v;
                var2[3] = south + 1;  // v_s²

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
    void naturalConvectionFromHeatedPlate() {

        int nx = 256;
        int ny = 256;

        float hx = 0.03f;
        float hy = 0.03f;

        double nu = 0.1;
        double alpha = 0.1;
        double beta = 1.0;
        double g = 1.0;

        double tHot = 1;
        double tAmbient = 0.5;
        double t0 = tAmbient; // buoyancy reference: undisturbed ambient fluid
        double deltaT = tHot - tAmbient;

        // Plate: 1 cell thick, centered horizontally and vertically,
        // spanning the middle 25% of the domain width.

        double plateWidth = 0.5;//meters

        int plateCells = (int) Math.round(plateWidth / hx);

        int plateCenter = nx / 2;

        int plateColStart = plateCenter - plateCells / 2;
        int plateColEnd   = plateCenter + plateCells / 2;
        plateColStart = Math.max(1, plateColStart);
        plateColEnd   = Math.min(nx - 2, plateColEnd);

        double pressurePenalty = 1e-3;

        double dt = 4;
        int nSteps = 1000;
        int printEvery = 4;
        int sampleEvery = 2;

        int cells = nx * ny;
        int n = cells * 4;

        Runtime rt = Runtime.getRuntime();

        System.out.printf("Max heap: %.2f GB%n",
                rt.maxMemory() / 1024.0 / 1024.0 / 1024.0);

        PaddedCSR3Tensor C = new PaddedCSR3Tensor(n, 4);
        PaddedCSRMatrix  A = new PaddedCSRMatrix(n, n, 8);

        double[] x = new double[n];
        double[] scaling = new double[n];
        double[] b = new double[n];

        DofRole[] role = new DofRole[n];
        double[] bcValue = new double[n];


        buildHeatedPlateOperator(nx, ny, hx, hy,nu, alpha, beta, g, tHot, tAmbient,
                plateCenter, plateColStart, plateColEnd, pressurePenalty, 1.0 / dt, C, A, scaling, role, bcValue);

        // Initial condition: fluid at rest, uniformly at ambient temperature
        // (Dirichlet dofs already hold their fixed values via bcValue).
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {

                int cell = j * nx + i;
                int u = 4 * cell, v = u + 1, T = u + 3;

                if (role[T] == DofRole.INTERIOR_T) x[T] = tAmbient;
                else if (role[T] == DofRole.DIRICHLET) x[T] = bcValue[T];

                if (role[u] == DofRole.DIRICHLET) x[u] = bcValue[u];
                if (role[v] == DofRole.DIRICHLET) x[v] = bcValue[v];
            }
        }

        double buoyancyConst = g * beta * t0;

        XYChart plumeChart = new XYChartBuilder()
                .width(900).height(400)
                .title("Plume strength")
                .xAxisTitle("t").yAxisTitle("max |velocity|")
                .build();
        plumeChart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);
        plumeChart.addSeries("max |velocity|", new double[]{0}, new double[]{0});

        LiveChartWindow plumeWindow = new LiveChartWindow(plumeChart);

        List<Double> tSampled        = new ArrayList<>();
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

            WorkingBuffer<DoubleVector> bufferN = new WorkingBuffer<>(7, () -> new CpuDoubleVector(n));
            WorkingBuffer<DoubleVector> bufferM = new WorkingBuffer<>(3, () -> new CpuDoubleVector(n));


            stats.reset();

            NewtonKrylov.solve(C, A, new CpuDoubleVector(x), new CpuDoubleVector(b), 100, 200,
                    1e-6 * nx * ny, 1e-7 * nx * ny, new CpuDoubleVector(scaling)
                    , stats, null, bufferN, bufferM);

            double t = step * dt;

            if (step % sampleEvery == 0 || step == nSteps - 1) {

                double maxSpeed = 0;
                for (int c = 0; c < cells; c++)
                    maxSpeed = Math.max(maxSpeed, Math.hypot(x[4 * c], x[4 * c + 1]));

                tSampled.add(t);
                maxSpeedSampled.add(maxSpeed);
            }

            if (step % printEvery == 0 || step == nSteps - 1) {


                plumeChart.updateXYSeries("max |velocity|", toArray(tSampled), toArray(maxSpeedSampled), null);
                plumeWindow.repaint();

                if (!plumeWindow.isInteractive())
                    plumeWindow.saveSnapshot("plume_latest.png");

                System.out.printf(java.util.Locale.ROOT,
                        "t=%.2f step=%d/%d newtonIters=%d linearIters=%d converged=%b%n",
                        t, step, nSteps, stats.newtonIterationCount, stats.linearIterations.size(),  stats.converged);

                double[][] speed = new double[nx][ny];
                double[][] temperature = new double[nx][ny];

                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++) {
                        int cell = j * nx + i;
                        int u = 4 * cell, v = u + 1, T = u + 3;
                        speed[i][j] = Math.hypot(x[u], x[v]);
                        temperature[i][j] = x[T];
                    }

                Field2DRenderer.saveHeatmap(speed, String.format("convection_plate/speed_%05d.png", step),
                        Field2DRenderer.Interpolation.NEAREST);
                Field2DRenderer.saveHeatmap(temperature, String.format("convection_plate/temp_%05d.png", step),
                        Field2DRenderer.Interpolation.NEAREST);
            }
        }
    }
}
