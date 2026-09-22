package com.rae.formicapi.math_tests.solvers;

import com.rae.formicapi.foundation.plotting.LiveChartWindow;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR3Tensor;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;
import com.rae.formicapi.foundation.math.solvers.NewtonKrylov;
import com.rae.formicapi.foundation.plotting.StreamlineRenderer;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.Styler;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.google.common.primitives.Doubles.toArray;

/**
 * Plain laminar incompressible flow past a cylinder in a channel, no
 * turbulence closure. Dropped after {@code CylinderVortexSheddingCompressibleTest}
 * for two reasons: SA's {@code cw1/d^2} destruction term put a near-singular
 * diagonal entry next to the near-wall cells (bad for the linear solve), and
 * the weakly-compressible pressure equation there needs an acoustic CFL
 * (`c*dt/h`) that isn't remotely satisfied at the timesteps we actually want
 * to take, so it rings every step instead of settling.
 *
 * State per cell, x = [u, v, p] -- back to the plain incompressible penalty
 * formulation ({@code Convection2Test}'s pattern): no wave equation, no
 * acoustic CFL, one less thing that can misbehave.
 *
 * Re = U_inf*D/nu = 100 (nu=0.1, D=10, U_inf=1) -- the textbook laminar
 * Karman-street case, well past the ~47 shedding-onset threshold and clean
 * periodic (St~0.165 by Williamson's correlation), so expected shedding
 * period ~ D/(St*U) ~ 60.7 time units.
 */
public class CylinderVortexSheddingLaminarTest {

    enum DofRole {
        INTERIOR,
        DIRICHLET, NEUMANN,
        RADIATION,   // reserved: future PML absorbing-layer boundary, not implemented yet
        FREE_OUTFLOW
    }

    static void buildCylinderOperator(int nx, int ny, double hx, double hy, double nu,
                                      double cylinderCenterX, double cylinderCenterY, double cylinderRadius,
                                      double uInf, double pInf, double pressurePenalty, double invDt,
                                      PaddedCSR3Tensor C, PaddedCSRMatrix A, double[] scaling, DofRole[] role, double[] bcValue) {

        double invHx = 1.0 / hx;
        double invHy = 1.0 / hy;
        double invHx2 = invHx * invHx;
        double invHy2 = invHy * invHy;

        double[] uRow = new double[7]; int[] uCols = new int[7];
        double[] vRow = new double[7]; int[] vCols = new int[7];
        double[] pRow = new double[5]; int[] pCols = new int[5];

        double[] cVal = new double[4]; int[] cVar1 = new int[4]; int[] cVar2 = new int[4];

        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {

                int cell = y * nx + x;
                int u = 3 * cell, v = u + 1, p = u + 2;

                double dxPhys = x * hx - cylinderCenterX;
                double dyPhys = y * hy - cylinderCenterY;
                double dCyl = Math.sqrt(Math.max(dxPhys * dxPhys, dyPhys * dyPhys)) - cylinderRadius;

                scaling[u] = 1;
                scaling[v] = 1;

                scaling[p] = 1/10.0;

                boolean isInlet    = x == 0;
                boolean isOutlet   = !isInlet && x == nx - 1;
                boolean isFarField = !isInlet && !isOutlet && (y == 0 || y == ny - 1);
                boolean isCylinder = !isInlet && !isOutlet && !isFarField && dCyl <= 0;

                if (isInlet) {
                    A.setRow(u, new double[]{1}, new int[]{u}, 1);
                    A.setRow(v, new double[]{1}, new int[]{v}, 1);
                    role[u] = DofRole.DIRICHLET; bcValue[u] = uInf;
                    role[v] = DofRole.DIRICHLET; bcValue[v] = 0;

                    int east = u + 3;
                    A.setRow(p, new double[]{1}, new int[]{p}, 1);
                    role[p] = DofRole.DIRICHLET;  bcValue[p] = pInf;

                    C.setRow(u, new double[0], new int[0], new int[0], 0);
                    C.setRow(v, new double[0], new int[0], new int[0], 0);
                    C.setRow(p, new double[0], new int[0], new int[0], 0);
                    continue;
                }

                if (isOutlet) {
                    // Zero-curvature ("do-nothing") outflow: d^2(field)/dx^2 = 0,
                    // i.e. the field extrapolates linearly from the two upstream
                    // interior cells, rather than the stiffer zero-gradient
                    // (d(field)/dx = 0) condition -- lets vortical structures
                    // leave the domain without reflecting as much.
                    int west = u - 3, westWest = u - 6;
                    A.setRow(u, new double[]{1, -2, 1}, new int[]{u, west, westWest}, 3);
                    A.setRow(v, new double[]{1, -2, 1}, new int[]{v, west + 1, westWest + 1}, 3);
                    A.setRow(p, new double[]{1, -2, 1}, new int[]{p, west + 2, westWest + 2}, 3);
                    role[u] = DofRole.FREE_OUTFLOW;
                    role[v] = DofRole.FREE_OUTFLOW;
                    role[p] = DofRole.FREE_OUTFLOW;

                    C.setRow(u, new double[0], new int[0], new int[0], 0);
                    C.setRow(v, new double[0], new int[0], new int[0], 0);
                    C.setRow(p, new double[0], new int[0], new int[0], 0);
                    continue;
                }
                if (isFarField) {
                    // Domain truncation approximating unconfined flow, not a
                    // wall: far from the body the flow is close to uniform
                    // freestream, so pin u,v to the freestream values, same
                    // as the inlet -- physically reasonable given the low
                    // blockage here (~9.5 diameters from the centerline to
                    // this boundary), and numerically robust, since a fixed
                    // Dirichlet value can't amplify an incoming disturbance
                    // the way an extrapolation-based condition can. Pressure
                    // stays Neumann (zero-gradient), same as the real inlet
                    // -- its level is set by the flow field, not prescribed.
                    A.setRow(u, new double[]{1}, new int[]{u}, 1);
                    A.setRow(v, new double[]{1}, new int[]{v}, 1);
                    role[u] = DofRole.DIRICHLET; bcValue[u] = uInf;
                    role[v] = DofRole.DIRICHLET; bcValue[v] = 0;

                    A.setRow(p, new double[]{1}, new int[]{p}, 1);
                    role[p] = DofRole.DIRICHLET; bcValue[p] = pInf;

                    C.setRow(u, new double[0], new int[0], new int[0], 0);
                    C.setRow(v, new double[0], new int[0], new int[0], 0);
                    C.setRow(p, new double[0], new int[0], new int[0], 0);

                    continue;
                }

                if (isCylinder) {
                    A.setRow(u, new double[]{1}, new int[]{u}, 1);
                    A.setRow(v, new double[]{1}, new int[]{v}, 1);
                    role[u] = DofRole.DIRICHLET; bcValue[u] = 0;
                    role[v] = DofRole.DIRICHLET; bcValue[v] = 0;

                    // Zero-gradient, not zero-value: pinning p=0 inside the
                    // solid fed a fake pressure jump straight into the
                    // adjacent fluid cell's momentum equation (its
                    // east/west/north/south pressure-gradient term reads
                    // exactly this cell). Extend p harmonically through the
                    // whole solid body instead -- the standard 5-point
                    // discrete Laplacian, 4*p_self - (p_e+p_w+p_n+p_s) = 0,
                    // applied uniformly regardless of whether a given
                    // neighbor happens to be fluid or solid (exact only for
                    // hx == hy, true everywhere in this file). Perimeter
                    // cells pick up real boundary data through whichever
                    // neighbors are fluid; interior cells just relay it
                    // inward, so the whole body ends up as a smooth harmonic
                    // extension of the surrounding pressure field rather
                    // than depending on an arbitrarily chosen single
                    // direction.
                    A.setRow(p, new double[]{4, -1, -1, -1, -1},
                            new int[]{p, p + 3, p - 3, p + 3 * nx, p - 3 * nx}, 5);
                    role[p] = DofRole.NEUMANN;

                    C.setRow(u, new double[0], new int[0], new int[0], 0);
                    C.setRow(v, new double[0], new int[0], new int[0], 0);
                    C.setRow(p, new double[0], new int[0], new int[0], 0);
                    continue;
                }

                // ---------------- fluid cell: plain laminar Navier-Stokes ----------------

                role[u] = DofRole.INTERIOR;
                role[v] = DofRole.INTERIOR;
                role[p] = DofRole.INTERIOR;

                int west = 3 * (cell - 1), east = 3 * (cell + 1);
                int south = 3 * (cell - nx), north = 3 * (cell + nx);

                uRow[0] = 2 * nu * (invHx2 + invHy2) + invDt;
                uRow[1] = -nu * invHx2; uRow[2] = -nu * invHx2;
                uRow[3] = -nu * invHy2; uRow[4] = -nu * invHy2;
                uRow[5] = 0.5 * invHx; uRow[6] = -0.5 * invHx;
                uCols[0] = u; uCols[1] = west; uCols[2] = east; uCols[3] = south; uCols[4] = north;
                uCols[5] = east + 2; uCols[6] = west + 2;
                A.setRow(u, uRow, uCols, 7);

                vRow[0] = 2 * nu * (invHx2 + invHy2) + invDt;
                vRow[1] = -nu * invHx2; vRow[2] = -nu * invHx2;
                vRow[3] = -nu * invHy2; vRow[4] = -nu * invHy2;
                vRow[5] = 0.5 * invHy; vRow[6] = -0.5 * invHy;
                vCols[0] = v; vCols[1] = west + 1; vCols[2] = east + 1; vCols[3] = south + 1; vCols[4] = north + 1;
                vCols[5] = north + 2; vCols[6] = south + 2;
                A.setRow(v, vRow, vCols, 7);

                pRow[0] = 0.5 * invHx; pRow[1] = -0.5 * invHx;
                pRow[2] = 0.5 * invHy; pRow[3] = -0.5 * invHy;
                pRow[4] = pressurePenalty + invDt;
                pCols[0] = east; pCols[1] = west; pCols[2] = north + 1; pCols[3] = south + 1; pCols[4] = p;
                A.setRow(p, pRow, pCols, 5);

                // Advection only -- no eddy-viscosity term, no mu.
                cVal[0] =  0.5 * invHx; cVar1[0] = u; cVar2[0] = east;
                cVal[1] = -0.5 * invHx; cVar1[1] = u; cVar2[1] = west;
                cVal[2] =  0.5 * invHy; cVar1[2] = u; cVar2[2] = north + 1;
                cVal[3] = -0.5 * invHy; cVar1[3] = u; cVar2[3] = south + 1;
                C.setRow(u, cVal, cVar1, cVar2, 4);

                cVal[0] =  0.5 * invHx; cVar1[0] = v; cVar2[0] = east + 1;
                cVal[1] = -0.5 * invHx; cVar1[1] = v; cVar2[1] = west + 1;
                cVal[2] =  0.5 * invHy; cVar1[2] = v; cVar2[2] = north + 1;
                cVal[3] = -0.5 * invHy; cVar1[3] = v; cVar2[3] = south + 1;
                C.setRow(v, cVal, cVar1, cVar2, 4);

                C.setRow(p, new double[0], new int[0], new int[0], 0);
            }
        }
    }

    //@Test
    void vortexSheddingBehindCylinder() {

        int nx = 400;
        int ny = 380;

        double hx = 1;
        double hy = 1;

        double D = 20.0;
        double R = D / 2.0;
        double cylinderCenterX = 5.0 * D;
        double cylinderCenterY = (ny - 1) * hy / 2.0;

        double uInf = 1.0;
        double pInf = 0;//1.013e5;
        double nu = 0.5;//uInf * D / Re;      // 0.1
        double Re = uInf * D/nu;              // textbook laminar Karman street

        double pressurePenalty = 1e-2;

        double dt = 0.1;
        double expectedStrouhal = 0.2684 - 1.0356 / Math.sqrt(Re); // Williamson's correlation
        double expectedPeriod = D / (expectedStrouhal * uInf);
        System.out.printf("Re=%.0f, expected St~%.3f, expected shedding period ~%.1f time units (%.0f steps)%n",
                Re, expectedStrouhal, expectedPeriod, expectedPeriod / dt);

        int nSteps = 10000;
        int printEvery = 10;
        int sampleEvery = 2;

        int cells = nx * ny;
        int n = cells * 3;

        Runtime rt = Runtime.getRuntime();
        System.out.printf("Max heap: %.2f GB%n", rt.maxMemory() / 1024.0 / 1024.0 / 1024.0);

        PaddedCSR3Tensor C = new PaddedCSR3Tensor(n, 4);
        PaddedCSRMatrix  A = new PaddedCSRMatrix(n, n, 7);

        double[] x = new double[n];
        double[] scaling = new double[n];
        double[] b = new double[n];

        DofRole[] role = new DofRole[n];
        double[] bcValue = new double[n];

        buildCylinderOperator(nx, ny, hx, hy, nu, cylinderCenterX, cylinderCenterY, R, uInf, pInf,
                pressurePenalty, 1.0 / dt, C, A, scaling, role, bcValue);

        // Initial condition: uniform freestream, plus a small antisymmetric v
        // perturbation just downstream of the cylinder so shedding doesn't
        // have to wait on floating-point noise to break the geometry's
        // symmetry.
        double bumpX = cylinderCenterX + 1.5 * D;
        Random rd = new Random();

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {

                int cell = j * nx + i;
                int u = 3 * cell, v = u + 1;

                if (role[u] == DofRole.INTERIOR) x[u] = uInf;
                else if (role[u] == DofRole.DIRICHLET) x[u] = bcValue[u];

                if (role[v] == DofRole.INTERIOR) {
                    double px = i * hx, py = j * hy;
                    x[v] = 0.02 * uInf * rd.nextDouble();//* Math.exp(-Math.pow((px - bumpX) / D, 2)) * ((py - cylinderCenterY) / R);
                } else if (role[v] == DofRole.DIRICHLET) x[v] = bcValue[v];
            }
        }

        // Wake probe: 4D downstream of the cylinder, at its centerline height.
        int probeI = (int) Math.round((cylinderCenterX + 4 * D) / hx);
        int probeJ = (int) Math.round(cylinderCenterY / hy);
        int probeCell = probeJ * nx + probeI;
        int probeV = 3 * probeCell + 1;

        XYChart wakeChart = new XYChartBuilder()
                .width(900).height(400)
                .title("Wake probe v-velocity (4D downstream)")
                .xAxisTitle("t").yAxisTitle("v").build();
        wakeChart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);
        wakeChart.addSeries("v(probe)", new double[]{0}, new double[]{0});

        LiveChartWindow wakeWindow = new LiveChartWindow(wakeChart);

        List<Double> tSampled = new ArrayList<>();
        List<Double> vSampled = new ArrayList<>();

        NewtonKrylov.Stats stats = new NewtonKrylov.Stats();

        double invHx = 1.0 / hx, invHy = 1.0 / hy;

        for (int step = 0; step < nSteps; step++) {

            for (int i = 0; i < n; i++) {
                switch (role[i]) {
                    case DIRICHLET -> b[i] = bcValue[i];
                    case NEUMANN, FREE_OUTFLOW -> b[i] = 0;
                    case RADIATION -> throw new UnsupportedOperationException("PML boundary not implemented yet");
                    case INTERIOR -> b[i] = x[i] / dt;
                }
            }

            WorkingBuffer<RealVector> bufferN = new WorkingBuffer<>(7, () -> new CpuDoubleVector(n));
            WorkingBuffer<RealVector> bufferM = new WorkingBuffer<>(3, () -> new CpuDoubleVector(n));

            stats.reset();

            NewtonKrylov.solve(C, A, new CpuDoubleVector(x), new CpuDoubleVector(b), 100, 200,
                    1e-6 * Math.sqrt(nx * ny), 1e-7 * Math.sqrt(nx * ny), new CpuDoubleVector(scaling),
                    stats, null, bufferN, bufferM);

            double t = step * dt;

            if (step % sampleEvery == 0 || step == nSteps - 1) {
                tSampled.add(t);
                vSampled.add(x[probeV]);
            }

            if (step % printEvery == 0 || step == nSteps - 1) {

                wakeChart.updateXYSeries("v(probe)", toArray(tSampled), toArray(vSampled), null);
                wakeWindow.repaint();

                if (!wakeWindow.isInteractive())
                    wakeWindow.saveSnapshot("cylinder_shedding_laminar_latest.png");

                int linearIt = 0;
                for (int i: stats.linearIterations){
                    linearIt += i;
                }
                System.out.printf(java.util.Locale.ROOT,
                        "t=%.2f step=%d/%d newtonIters=%d linearIters=%d converged=%b v(probe)=%.4f%n",
                        t, step, nSteps, stats.newtonIterationCount, linearIt,
                        stats.converged, x[probeV]);

                double[][] speed = new double[nx][ny];
                double[][] u = new double[nx][ny];
                double[][] v = new double[nx][ny];
                double[][] vorticity = new double[nx][ny];
                double[][] pressure = new double[nx][ny];

                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++) {
                        int cell = j * nx + i;
                        int iu = 3 * cell, iv = iu + 1, p = iv + 1;
                        speed[i][j] = Math.hypot(x[iu], x[iv]);
                        u[i][j] = x[iu];
                        v[i][j] = x[iv];
                        pressure[i][j] = x[p];
                    }

                for (int j = 1; j < ny - 1; j++)
                    for (int i = 1; i < nx - 1; i++) {
                        int cell = j * nx + i;
                        int east = 3 * (cell + 1), west = 3 * (cell - 1);
                        int north = 3 * (cell + nx), south = 3 * (cell - nx);
                        double dvdx = (x[east + 1] - x[west + 1]) * 0.5 * invHx;
                        double dudy = (x[north] - x[south]) * 0.5 * invHy;
                        vorticity[i][j] = dvdx - dudy;
                    }

                com.rae.formicapi.foundation.plotting.Field2DRenderer.saveHeatmap(speed, String.format("cylinder_shedding_laminar/speed/%05d.png", step),
                        com.rae.formicapi.foundation.plotting.Field2DRenderer.Interpolation.NEAREST);
                com.rae.formicapi.foundation.plotting.Field2DRenderer.saveHeatmap(vorticity, String.format("cylinder_shedding_laminar/vorticity/%05d.png", step),
                        com.rae.formicapi.foundation.plotting.Field2DRenderer.Interpolation.NEAREST);
                com.rae.formicapi.foundation.plotting.Field2DRenderer.saveHeatmap(pressure, String.format("cylinder_shedding_laminar/pressure/%05d.png", step),
                        com.rae.formicapi.foundation.plotting.Field2DRenderer.Interpolation.NEAREST);
                StreamlineRenderer.saveStreamlines( u, v, String.format("cylinder_shedding_laminar/streamline/%05d.png", step));
            }
        }
    }
}