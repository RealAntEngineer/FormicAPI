package com.rae.formicapi.math_tests.solvers;

import com.rae.formicapi.LiveChartWindow;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR3Tensor;
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

/**
 * URANS + simplified Spalart-Allmaras flow past a cylinder, in a channel
 * (no-slip top/bottom, matching {@code Convection2Test}'s outer-wall
 * treatment rather than inventing a new free-slip BC). No temperature yet --
 * this is purely to exercise the quadratic-tensor machinery on a genuinely
 * coupled turbulence field and check {@link NewtonKrylov}'s behavior on a
 * larger, oscillatory problem.
 *
 * State per cell, x = [u, v, p, mu] (mu is the SA working variable
 * "nu-tilde"; mu_t = Cv * mu, no fv1 blending -- same simplification as
 * discussed for the general URANS/SA design).
 *
 * <h2>Simplifications relative to real SA (in addition to mu_t = Cv*mu)</h2>
 * <ul>
 *   <li>Production uses <b>signed</b> vorticity Omega = dv/dx - du/dy, not
 *       the magnitude |S~|. Real SA uses the magnitude specifically to keep
 *       production non-negative; that's not expressible as a bilinear
 *       (quadratic-tensor) term, so it's dropped, same spirit as skipping
 *       fv1/fv2/fv3.</li>
 *   <li>The cb2*|grad(mu)|^2 cross-diffusion source is dropped entirely
 *       (minor term, not needed to see the machinery work). cw1 still uses
 *       its standard value.</li>
 *   <li>Wall distance is the closed-form {@code min(distance to cylinder,
 *       distance to bottom wall, distance to top wall)} -- exact for a
 *       circle in a channel, no distance-transform needed.</li>
 *   <li>Advection/production use the same non-conservative
 *       "self * neighbor-difference" discretization Convection2Test already
 *       uses for u.grad(u) and u.grad(T) -- not re-derived, just reused for
 *       u.grad(mu) and mu*Omega.</li>
 * </ul>
 *
 * <h2>Regime</h2>
 * Re = U_inf*D/nu = 10000 (turbulent-regime vortex shedding, Strouhal
 * ~0.2, well past the ~47 laminar-shedding-onset threshold). The grid gives
 * roughly 20 cells across the diameter -- nowhere near enough to resolve a
 * real turbulent boundary layer, but that's fine here: the goal is checking
 * that the solver produces a stable, oscillating wake, not a
 * publication-quality Strouhal number.
 */
public class CylinderVortexSheddingCompressibleTest {

    private static final double PRESSURE_DIFFUSION = 0.1;

    enum DofRole {
        INTERIOR_U, INTERIOR_V, INTERIOR_P, INTERIOR_MU,
        DIRICHLET,
        NEUMANN
    }

    static void buildCylinderOperator(int nx, int ny, double hx, double hy, double nu,
                                      double cylinderCenterX, double cylinderCenterY, double cylinderRadius,
                                      double uInf, double muInf,
                                      double cb1, double cw1, double sigma, double cv,
                                      double rho0, double soundSpeed, double invDt,
                                      PaddedCSR3Tensor C, PaddedCSRMatrix A, double[] scaling, DofRole[] role,
                                      double[] bcValue, double[] wallDistance) {

        double invHx = 1.0 / hx;
        double invHy = 1.0 / hy;
        double invHx2 = invHx * invHx;
        double invHy2 = invHy * invHy;
        double epsD = 1 * Math.min(hx, hy);//0.05 * Math.min(hx, hy);

        double[] uRow = new double[7]; int[] uCols = new int[7];
        double[] vRow = new double[7]; int[] vCols = new int[7];
        double[] pRow = new double[5]; int[] pCols = new int[5];
        double[] mRow = new double[5]; int[] mCols = new int[5];

        // Shared C-row scratch, sized for the largest row (mu: 13 entries).
        double[] cVal = new double[13];
        int[] cVar1 = new int[13];
        int[] cVar2 = new int[13];

        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {

                int cell = y * nx + x;
                int u = 4 * cell, v = u + 1, p = u + 2, m = u + 3;

                double dxPhys = x * hx - cylinderCenterX;
                double dyPhys = y * hy - cylinderCenterY;
                double dCyl = Math.sqrt(dxPhys * dxPhys + dyPhys * dyPhys) - cylinderRadius;
                double dBot = y * hy;
                double dTop = (ny - 1 - y) * hy;
                wallDistance[cell] = Math.max(epsD, Math.min(dCyl, Math.min(dBot, dTop)));

                scaling[u] = 1;
                scaling[v] = 1;
                scaling[p] = 1.0;
                scaling[m] = 1/10.0;

                boolean isInlet    = x == 0;
                boolean isOutlet   = !isInlet && x == nx - 1;
                boolean isWall     = !isInlet && !isOutlet && (y == 0 || y == ny - 1);
                boolean isCylinder = !isInlet && !isOutlet && !isWall && dCyl <= 0;

                if (isInlet) {
                    A.setRow(u, new double[]{1}, new int[]{u}, 1);
                    A.setRow(v, new double[]{1}, new int[]{v}, 1);
                    A.setRow(m, new double[]{1}, new int[]{m}, 1);
                    role[u] = DofRole.DIRICHLET; bcValue[u] = uInf;
                    role[v] = DofRole.DIRICHLET; bcValue[v] = 0;
                    role[m] = DofRole.DIRICHLET; bcValue[m] = muInf;

                    int east = u + 4;
                    A.setRow(p, new double[]{1, -1}, new int[]{p, east + 2}, 2);
                    role[p] = DofRole.NEUMANN;

                    C.setRow(u, new double[0], new int[0], new int[0], 0);
                    C.setRow(v, new double[0], new int[0], new int[0], 0);
                    C.setRow(p, new double[0], new int[0], new int[0], 0);
                    C.setRow(m, new double[0], new int[0], new int[0], 0);
                    continue;
                }

                if (isOutlet) {
                    int west = u - 4;
                    A.setRow(u, new double[]{1, -1}, new int[]{u, west}, 2);
                    A.setRow(v, new double[]{1, -1}, new int[]{v, west + 1}, 2);
                    A.setRow(m, new double[]{1, -1}, new int[]{m, west + 3}, 2);
                    A.setRow(p, new double[]{1, -1}, new int[]{p, west + 2}, 2);
                    role[u] = DofRole.NEUMANN;
                    role[v] = DofRole.NEUMANN;
                    role[m] = DofRole.NEUMANN;
                    role[p] = DofRole.NEUMANN;

                    C.setRow(u, new double[0], new int[0], new int[0], 0);
                    C.setRow(v, new double[0], new int[0], new int[0], 0);
                    C.setRow(p, new double[0], new int[0], new int[0], 0);
                    C.setRow(m, new double[0], new int[0], new int[0], 0);
                    continue;
                }

                if (isWall || isCylinder) {
                    A.setRow(u, new double[]{1}, new int[]{u}, 1);
                    A.setRow(v, new double[]{1}, new int[]{v}, 1);
                    A.setRow(m, new double[]{1}, new int[]{m}, 1);
                    role[u] = DofRole.DIRICHLET; bcValue[u] = 0;
                    role[v] = DofRole.DIRICHLET; bcValue[v] = 0;
                    role[m] = DofRole.DIRICHLET; bcValue[m] = 0; // standard SA wall condition

                    if (isCylinder) {
                        // Decoupled, same trick as Convection2Test's isPlate:
                        // continuity never reads a solid neighbor's p, only its u/v.
                        /*A.setRow(p, new double[]{1}, new int[]{p}, 1);
                        role[p] = DofRole.DIRICHLET;
                        bcValue[p] = 0;*/
                        int pNeighbor;

                        // Pick the fluid-side neighbor approximately along the radial direction.
                        // For the first/simple version, use whichever Cartesian direction points
                        // most strongly toward the cylinder center.
                        double ax = Math.abs(dxPhys);
                        double ay = Math.abs(dyPhys);

                        if (ax > ay) {
                            pNeighbor = dxPhys > 0 ? p - 4 : p + 4;
                        } else {
                            pNeighbor = dyPhys > 0 ? p - 4 * nx : p + 4 * nx;
                        }

                        A.setRow(p,
                                new double[]{1, -1},
                                new int[]{p, pNeighbor},
                                2);

                        role[p] = DofRole.NEUMANN;
                    } else {
                        int pNeighbor = (y == 0) ? p + 4 * nx : p - 4 * nx;
                        A.setRow(p, new double[]{1, -1}, new int[]{p, pNeighbor}, 2);
                        role[p] = DofRole.NEUMANN;
                    }

                    C.setRow(u, new double[0], new int[0], new int[0], 0);
                    C.setRow(v, new double[0], new int[0], new int[0], 0);
                    C.setRow(p, new double[0], new int[0], new int[0], 0);
                    C.setRow(m, new double[0], new int[0], new int[0], 0);
                    continue;
                }

                // ---------------- fluid cell: full coupled URANS/SA ----------------

                role[u] = DofRole.INTERIOR_U;
                role[v] = DofRole.INTERIOR_V;
                role[p] = DofRole.INTERIOR_P;
                role[m] = DofRole.INTERIOR_MU;

                int west = 4 * (cell - 1), east = 4 * (cell + 1);
                int south = 4 * (cell - nx), north = 4 * (cell + nx);

                // --- linear (molecular-viscosity-only) parts: A ---

                uRow[0] = 2 * nu * (invHx2 + invHy2) + invDt;
                uRow[1] = -nu * invHx2; uRow[2] = -nu * invHx2;
                uRow[3] = -nu * invHy2; uRow[4] = -nu * invHy2;
                uRow[5] = 0.5 * invHx / rho0; uRow[6] = -0.5 * invHx / rho0;
                uCols[0] = u; uCols[1] = west; uCols[2] = east; uCols[3] = south; uCols[4] = north;
                uCols[5] = east + 2; uCols[6] = west + 2;
                A.setRow(u, uRow, uCols, 7);

                vRow[0] = 2 * nu * (invHx2 + invHy2) + invDt;
                vRow[1] = -nu * invHx2; vRow[2] = -nu * invHx2;
                vRow[3] = -nu * invHy2; vRow[4] = -nu * invHy2;
                vRow[5] = 0.5 * invHy / rho0; vRow[6] = -0.5 * invHy / rho0;
                vCols[0] = v; vCols[1] = west + 1; vCols[2] = east + 1; vCols[3] = south + 1; vCols[4] = north + 1;
                vCols[5] = north + 2; vCols[6] = south + 2;
                A.setRow(v, vRow, vCols, 7);

                pRow[0] = 0.5 * invHx; pRow[1] = -0.5 * invHx;
                pRow[2] = 0.5 * invHy; pRow[3] = -0.5 * invHy;
                //pRow[4] = pressurePenalty;
                double pressureDt = 1.0 / (rho0 * soundSpeed * soundSpeed) * invDt;

                //double pressureDiffusion = PRESSURE_DIFFUSION * hx * hx * pressureDt;

                pRow[4] = pressureDt; //+ 2.0 * pressureDiffusion * (invHx2 + invHy2);//pressureDt + 2.0 * pressureDiffusion * (invHx2 + invHy2);

                pCols[0] = east; pCols[1] = west; pCols[2] = north + 1; pCols[3] = south + 1; pCols[4] = p;
                A.setRow(p, pRow, pCols, 5);

                double nuOverSigma = nu / sigma;
                mRow[0] = 2 * nuOverSigma * (invHx2 + invHy2) + invDt;
                mRow[1] = -nuOverSigma * invHx2; mRow[2] = -nuOverSigma * invHx2;
                mRow[3] = -nuOverSigma * invHy2; mRow[4] = -nuOverSigma * invHy2;
                mCols[0] = m; mCols[1] = west + 3; mCols[2] = east + 3; mCols[3] = south + 3; mCols[4] = north + 3;
                A.setRow(m, mRow, mCols, 5);

                // --- quadratic parts: C ---

                double d = wallDistance[cell];

                // u-momentum: advection u.grad(u) (Convection2Test's non-conservative
                // "self * neighbor-difference" form) + eddy diffusion Cv*mu*Laplacian(u).
                cVal[0] =  0.5 * invHx; cVar1[0] = u; cVar2[0] = east;
                cVal[1] = -0.5 * invHx; cVar1[1] = u; cVar2[1] = west;
                cVal[2] =  0.5 * invHy; cVar1[2] = u; cVar2[2] = north + 1;
                cVal[3] = -0.5 * invHy; cVar1[3] = u; cVar2[3] = south + 1;
                cVal[4] =  2 * cv * (invHx2 + invHy2); cVar1[4] = m; cVar2[4] = u;
                cVal[5] = -cv * invHx2; cVar1[5] = m; cVar2[5] = east;
                cVal[6] = -cv * invHx2; cVar1[6] = m; cVar2[6] = west;
                cVal[7] = -cv * invHy2; cVar1[7] = m; cVar2[7] = north;
                cVal[8] = -cv * invHy2; cVar1[8] = m; cVar2[8] = south;
                C.setRow(u, cVal, cVar1, cVar2, 9);

                // v-momentum: same shape, no buoyancy term this time (no T).
                cVal[0] =  0.5 * invHx; cVar1[0] = v; cVar2[0] = east + 1;
                cVal[1] = -0.5 * invHx; cVar1[1] = v; cVar2[1] = west + 1;
                cVal[2] =  0.5 * invHy; cVar1[2] = v; cVar2[2] = north + 1;
                cVal[3] = -0.5 * invHy; cVar1[3] = v; cVar2[3] = south + 1;
                cVal[4] =  2 * cv * (invHx2 + invHy2); cVar1[4] = m; cVar2[4] = v;
                cVal[5] = -cv * invHx2; cVar1[5] = m; cVar2[5] = east + 1;
                cVal[6] = -cv * invHx2; cVar1[6] = m; cVar2[6] = west + 1;
                cVal[7] = -cv * invHy2; cVar1[7] = m; cVar2[7] = north + 1;
                cVal[8] = -cv * invHy2; cVar1[8] = m; cVar2[8] = south + 1;
                C.setRow(v, cVal, cVar1, cVar2, 9);

                C.setRow(p, new double[0], new int[0], new int[0], 0);

                // SA transport: advection u.grad(mu) + production(-cb1*Omega*mu, signed
                // vorticity) + [quadratic diffusion mu*Laplacian(mu), diagonal combined
                // with the destruction term cw1*(mu/d)^2].
                cVal[0] =  0.5 * invHx; cVar1[0] = u; cVar2[0] = east + 3;
                cVal[1] = -0.5 * invHx; cVar1[1] = u; cVar2[1] = west + 3;
                cVal[2] =  0.5 * invHy; cVar1[2] = v; cVar2[2] = north + 3;
                cVal[3] = -0.5 * invHy; cVar1[3] = v; cVar2[3] = south + 3;

                cVal[4] = -cb1 * 0.5 * invHx; cVar1[4] = m; cVar2[4] = east + 1;  // -cb1*mu*dv/dx
                cVal[5] =  cb1 * 0.5 * invHx; cVar1[5] = m; cVar2[5] = west + 1;
                cVal[6] =  cb1 * 0.5 * invHy; cVar1[6] = m; cVar2[6] = north;     // +cb1*mu*du/dy
                cVal[7] = -cb1 * 0.5 * invHy; cVar1[7] = m; cVar2[7] = south;

                cVal[8]  = cw1 / (d * d) + (2.0 / sigma) * (invHx2 + invHy2); cVar1[8] = m; cVar2[8] = m;
                cVal[9]  = -invHx2 / sigma; cVar1[9]  = m; cVar2[9]  = east + 3;
                cVal[10] = -invHx2 / sigma; cVar1[10] = m; cVar2[10] = west + 3;
                cVal[11] = -invHy2 / sigma; cVar1[11] = m; cVar2[11] = north + 3;
                cVal[12] = -invHy2 / sigma; cVar1[12] = m; cVar2[12] = south + 3;

                C.setRow(m, cVal, cVar1, cVar2, 13);
            }
        }
    }

    //TODO add parallelisation
    //@Test
    void vortexSheddingBehindCylinder() {

        int nx = 200;
        int ny = 140;

        double hx = 1;
        double hy = 1;

        double D = 10.0;
        double R = D / 2.0;
        double cylinderCenterX = 3.0 * D;              // 5D from the inlet
        double cylinderCenterY = (ny - 1) * hy / 2.0;

        double uInf = 1.0;
        double nu = 1e-1;//uInf * D / Re;                  // 1e-4
        double muInf = 30;                     // standard SA freestream value ~3*nu

        double cv = 1.0;                              // mu_t = Cv * mu
        double cb1 = 0.1355;
        double sigma = 2.0 / 3.0;
        double cb2 = 0.622;                           // only used to derive the standard cw1
        double kappa = 0.41;
        double cw1 = cb1 / (kappa * kappa) + (1 + cb2) / sigma;

        //double pressurePenalty = 1e-2;

        double rho0 = 1.25;
        double soundSpeed = 30.0;

        double dt = 0.4;
        int nSteps = 1000;                             // ~10 shedding periods at St~0.2; extend for more
        int printEvery = 10;
        int sampleEvery = 2;

        int cells = nx * ny;
        int n = cells * 4;

        Runtime rt = Runtime.getRuntime();
        System.out.printf("Max heap: %.2f GB%n", rt.maxMemory() / 1024.0 / 1024.0 / 1024.0);

        PaddedCSR3Tensor C = new PaddedCSR3Tensor(n, 13);
        PaddedCSRMatrix  A = new PaddedCSRMatrix(n, n, 7);

        double[] x = new double[n];
        double[] scaling = new double[n];
        double[] b = new double[n];
        double[] wallDistance = new double[cells];

        DofRole[] role = new DofRole[n];
        double[] bcValue = new double[n];

        buildCylinderOperator(nx, ny, hx, hy, nu, cylinderCenterX, cylinderCenterY, R, uInf, muInf,
                cb1, cw1, sigma, cv, rho0, soundSpeed, 1.0 / dt, C, A, scaling, role, bcValue, wallDistance);

        // Initial condition: uniform freestream, at rest boundary-wise
        // (Dirichlet dofs already hold their fixed values via bcValue), plus a
        // small antisymmetric v perturbation just downstream of the cylinder
        // so shedding doesn't have to wait on floating-point noise to break
        // the geometry's symmetry.
        double bumpX = cylinderCenterX + 1.5 * D;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {

                int cell = j * nx + i;
                int u = 4 * cell, v = u + 1, m = u + 3;

                if (role[u] == DofRole.INTERIOR_U) x[u] = uInf;
                else if (role[u] == DofRole.DIRICHLET) x[u] = bcValue[u];

                if (role[v] == DofRole.INTERIOR_V) {
                    double px = i * hx, py = j * hy;
                    x[v] = 0.02 * uInf * Math.exp(-Math.pow((px - bumpX) / D, 2)) * ((py - cylinderCenterY) / R);
                } else if (role[v] == DofRole.DIRICHLET) x[v] = bcValue[v];

                if (role[m] == DofRole.INTERIOR_MU) x[m] = muInf;
                else if (role[m] == DofRole.DIRICHLET) x[m] = bcValue[m];
            }
        }

        // Wake probe: 4D downstream of the cylinder, at its centerline height.
        // A clean oscillating v(t) here is the real "did shedding happen" check
        // -- more telling than the Newton/linear residuals alone.
        int probeI = (int) Math.round((cylinderCenterX + 4 * D) / hx);
        int probeJ = (int) Math.round(cylinderCenterY / hy);
        int probeCell = probeJ * nx + probeI;
        int probeV = 4 * probeCell + 1;

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

        double pressureDt = 1.0 / (rho0 * soundSpeed * soundSpeed * dt);

        for (int step = 0; step < nSteps; step++) {

            for (int i = 0; i < n; i++) {
                switch (role[i]) {
                    case DIRICHLET -> b[i] = bcValue[i];
                    case NEUMANN -> b[i] = 0;
                    case INTERIOR_P -> b[i] = pressureDt * x[i];
                    case INTERIOR_U, INTERIOR_V, INTERIOR_MU -> b[i] = x[i] / dt;
                }
            }

            WorkingBuffer<DoubleVector> bufferN = new WorkingBuffer<>(7, () -> new CpuDoubleVector(n));
            WorkingBuffer<DoubleVector> bufferM = new WorkingBuffer<>(3, () -> new CpuDoubleVector(n));

            stats.reset();

            // Unconstrained/square, same as Convection2Test: every boundary
            // dof already has a trivial row baked into A/C, no row reduction.
            // See the scaling discussion from Convection2Test before trusting
            // these tolerances/scaling as-is -- worth watching the wake probe
            // signal, not just stats.converged, while tuning.
            NewtonKrylov.solve(C, A, new CpuDoubleVector(x), new CpuDoubleVector(b), 100, 200,
                    1e-6 * nx * ny, 1e-8 * nx * ny, new CpuDoubleVector(scaling),
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
                    wakeWindow.saveSnapshot("cylinder_shedding_latest.png");

                System.out.printf(java.util.Locale.ROOT,
                        "t=%.2f step=%d/%d newtonIters=%d linearIters=%d converged=%b v(probe)=%.4f%n",
                        t, step, nSteps, stats.newtonIterationCount, stats.linearIterations.size(),
                        stats.converged, x[probeV]);

                double[][] speed = new double[nx][ny];
                double[][] vorticity = new double[nx][ny];
                double[][] pressure = new double[nx][ny];
                double[][] mu = new double[nx][ny];

                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++) {
                        int cell = j * nx + i;
                        int u = 4 * cell, v = u + 1, p = v + 1, imu = p + 1;
                        speed[i][j] = Math.hypot(x[u], x[v]);
                        pressure[i][j] = x[p];
                        mu[i][j] = x[imu];
                    }

                for (int j = 1; j < ny - 1; j++)
                    for (int i = 1; i < nx - 1; i++) {
                        int cell = j * nx + i;
                        int east = 4 * (cell + 1), west = 4 * (cell - 1);
                        int north = 4 * (cell + nx), south = 4 * (cell - nx);
                        double dvdx = (x[east + 1] - x[west + 1]) * 0.5 * invHx;
                        double dudy = (x[north] - x[south]) * 0.5 * invHy;
                        vorticity[i][j] = dvdx - dudy;
                    }

                Field2DRenderer.saveHeatmap(speed, String.format("cylinder_shedding_compressible/speed/%05d.png", step),
                        Field2DRenderer.Interpolation.NEAREST);
                Field2DRenderer.saveHeatmap(vorticity, String.format("cylinder_shedding_compressible/vorticity/%05d.png", step),
                        Field2DRenderer.Interpolation.NEAREST);
                Field2DRenderer.saveHeatmap(pressure, String.format("cylinder_shedding_compressible/pressure/%05d.png", step),
                        Field2DRenderer.Interpolation.NEAREST);
                Field2DRenderer.saveHeatmap(mu, String.format("cylinder_shedding_compressible/mu/%05d.png", step),
                        Field2DRenderer.Interpolation.NEAREST);
            }
        }
    }
}