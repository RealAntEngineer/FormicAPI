package com.rae.formicapi.math_tests.lagrangian;

import com.rae.formicapi.foundation.math.lagrangian.ParticleSystem;
import com.rae.formicapi.foundation.math.lagrangian.RK2Advector;
import com.rae.formicapi.foundation.math.lagrangian.VelocityResampler;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuBooleanVector;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuIntegerVector;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;
import com.rae.formicapi.foundation.math.solvers.BiCGStab;
// ASSUMPTION: adjust to wherever these actually live in your tree -- same
// three renderers BentPlateVorteLaminarTest uses (Field2DRenderer,
// StreamlineRenderer, VectorFieldRenderer). Not shown in the excerpt I
// have, so this package name is a guess.
import com.rae.formicapi.foundation.plotting.Field2DRenderer;
import com.rae.formicapi.foundation.plotting.StreamlineRenderer;
import com.rae.formicapi.foundation.plotting.VectorFieldRenderer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * 2D toy test: buoyant PIC flow past a ground-mounted obstacle, on a
 * coarse grid (1 m cells) with a large outer timestep (1 s) -- the regime
 * a purely Eulerian advection scheme struggles with in detached,
 * recirculating flow. This is the Java/lagrangian-package counterpart of
 * the earlier Python PIC toy.
 *
 * <p>Exercises the full stack end to end:
 * <ul>
 *   <li>{@link ParticleSystem} spawn / kill / compact lifecycle</li>
 *   <li>{@link RK2Advector}'s grid-coupled G2P (via {@link VelocityResampler})
 *       plus its particle-force integration path</li>
 *   <li>{@link BiCGStab}, reused completely unchanged, for the pressure
 *       projection -- the one elliptic solve per step that enforces
 *       incompressibility</li>
 *   <li>P2G / BC enforcement done entirely with existing {@link RealVector}
 *       primitives ({@code skippedAxpy}, {@code gather}, {@code skippedScale}),
 *       no new API needed for this test</li>
 * </ul>
 *
 * <p>Everything physics-specific (obstacle geometry, fixed temperature
 * field, buoyancy law, boundary conditions) lives in this test file, not
 * in the lagrangian package -- {@link ParticleSystem}/{@link RK2Advector}
 * never see a temperature, an obstacle, or a boundary condition, only
 * {@link RealVector}s and a {@link VelocityResampler} callback.
 *
 * <p><b>Design note on buoyancy placement.</b> Buoyancy is applied via
 * {@code particles.force(1)} inside the {@link VelocityResampler}
 * (sampling the fixed temperature field at the particle's current
 * position each RK2 evaluation), specifically to exercise
 * {@link RK2Advector}'s force-integration path. This means buoyancy added
 * mid-step is not re-projected until the *next* outer step's pressure
 * solve -- a small transient divergence can appear within a step and gets
 * cleaned up at the following projection. For a fully divergence-free
 * treatment at every instant, add buoyancy to the grid {@code v} field
 * before projection instead (as the Python toy does); this test
 * deliberately takes the particle-force route to validate that path.
 */
public class BuoyantWakePicTest {

    // ---- domain: coarse cells, large timestep -------------------------
    static final int nx = 240, ny = 160;
    static final double dx = 1.0, dy = 1.0;
    static final double Lx = nx * dx, Ly = ny * dy;
    static final double dt = 1.0;
    static final double rho = 1.2;
    static final double g = 9.81;
    static final double T0 = 293.0;
    static final double U0 = 2.0;
    static final int nSteps = 40;
    static final int subSteps = 4;
    static final int perCell = 4;

    // ---- obstacle geometry: single source of truth, in cells ----------
    // (i0,i1] x [j0,j1] inclusive on both ends, ground-mounted.
    // Domain and obstacle both scaled 10x from the original toy (24x16
    // domain, 3x5 obstacle) -- same relative geometry, 100x more cells.
    static final int obstacleI0 = 80, obstacleI1 = 100;   // 30 cells wide
    static final int obstacleJ0 = 0, obstacleJ1 = 40;     // 50 cells tall
    static final double hotPatchCenterX = (obstacleI0 + obstacleI1) / 2.0 * dx + 30.0; // just downwind of the wake

    // u-faces: (nx+1) x ny at x=i*dx, y=(j+0.5)*dy  -- flattened i*ny+j
    // v-faces: nx x (ny+1) at x=(i+0.5)*dx, y=j*dy  -- flattened i*(ny+1)+j
    static int uN() { return (nx + 1) * ny; }
    static int vN() { return nx * (ny + 1); }
    static int uIdx(int i, int j) { return i * ny + j; }
    static int vIdx(int i, int j) { return i * (ny + 1) + j; }

    static boolean solidCell(int i, int j) {
        return i >= obstacleI0 && i <= obstacleI1 && j >= obstacleJ0 && j <= obstacleJ1;
    }

    static double inflowU(double y) {
        return U0 * (0.4 + 0.6 * Math.min(1.0, Math.max(0.0, y / Ly)));
    }

    /** Fixed temperature field, stand-in for another simulation's output. */
    static double temperatureAt(double x, double y) {
        double ddx = x - hotPatchCenterX;
        return T0 + 20.0 * Math.exp(-(ddx * ddx) / (2 * 25.0 * 25.0)) * Math.exp(-y / 15.0);
    }

    static double temperatureCellCenter(int i, int j) {
        return temperatureAt((i + 0.5) * dx, (j + 0.5) * dy);
    }

    // ------------------------------------------------------------------
    // Bilinear weight quads: fundamentally a per-particle floor/clip
    // computation, so (consistent with how BentPlateVorteLaminarTest
    // builds its per-cell CSR rows via plain arrays) this is done as a
    // direct Java loop over raw positions, not forced into DoubleVector
    // primitives. Everything AFTER this -- the actual scatter/gather --
    // uses only DoubleVector ops.
    // ------------------------------------------------------------------
    private static final class CornerWeights {
        IntegerVector c00, c10, c01, c11;
        RealVector w00, w10, w01, w11;
    }

    private static CornerWeights bilinearWeights(double[] px, double[] py, int n,
                                                 int Nx, int Ny, double offx, double offy,
                                                 java.util.function.IntBinaryOperator flatten) {
        CornerWeights cw = new CornerWeights();
        int[] c00 = new int[n], c10 = new int[n], c01 = new int[n], c11 = new int[n];
        double[] w00 = new double[n], w10 = new double[n], w01 = new double[n], w11 = new double[n];
        for (int k = 0; k < n; k++) {
            double fx = px[k] / dx - offx;
            double fy = py[k] / dy - offy;
            int i0 = clampInt((int) Math.floor(fx), 0, Nx - 2);
            int j0 = clampInt((int) Math.floor(fy), 0, Ny - 2);
            int i1 = i0 + 1, j1 = j0 + 1;
            double tx = clampD(fx - i0, 0, 1);
            double ty = clampD(fy - j0, 0, 1);
            c00[k] = flatten.applyAsInt(i0, j0); w00[k] = (1 - tx) * (1 - ty);
            c10[k] = flatten.applyAsInt(i1, j0); w10[k] = tx * (1 - ty);
            c01[k] = flatten.applyAsInt(i0, j1); w01[k] = (1 - tx) * ty;
            c11[k] = flatten.applyAsInt(i1, j1); w11[k] = tx * ty;
        }
        cw.c00 = new CpuIntegerVector(n); cw.c10 = new CpuIntegerVector(n);
        cw.c01 = new CpuIntegerVector(n); cw.c11 = new CpuIntegerVector(n);
        for (int k = 0; k < n; k++) {
            cw.c00.set(c00[k], k); cw.c10.set(c10[k], k);
            cw.c01.set(c01[k], k); cw.c11.set(c11[k], k);
        }
        cw.w00 = new CpuDoubleVector(w00); cw.w10 = new CpuDoubleVector(w10);
        cw.w01 = new CpuDoubleVector(w01); cw.w11 = new CpuDoubleVector(w11);
        return cw;
    }

    private static int clampInt(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }
    private static double clampD(double v, double lo, double hi) { return Math.max(lo, Math.min(hi, v)); }

    /** P2G: scatter a particle-space DoubleVector onto a grid-space field of size {@code gridN}. */
    private static RealVector scatterToGrid(RealVector particleValues, int n, int gridN, CornerWeights cw) {
        RealVector      accum  = new CpuDoubleVector(gridN);
        RealVector      weight = new CpuDoubleVector(gridN);
        RealVector[]    w      = {cw.w00, cw.w10, cw.w01, cw.w11};
        IntegerVector[] c      = {cw.c00, cw.c10, cw.c01, cw.c11};
        for (int corner = 0; corner < 4; corner++) {
            RealVector temp = (RealVector) particleValues.copy();
            temp.scale(w[corner]);
            accum.skippedAxpy(1.0, temp, c[corner], true, false);
            weight.skippedAxpy(1.0, w[corner], c[corner], true, false);
        }
        weight.add(1e-12); // avoids 0/0 -> NaN at untouched nodes; accum is 0 there too, so result is 0
        accum.divide(weight);
        return accum;
    }

    /** G2P: gather a grid-space field into a particle-space DoubleVector of size {@code n}, overwriting {@code out}. */
    private static void gatherToParticles(RealVector grid, RealVector out, int n, CornerWeights cw) {
        out.clear();
        RealVector[]    w    = {cw.w00, cw.w10, cw.w01, cw.w11};
        IntegerVector[] c    = {cw.c00, cw.c10, cw.c01, cw.c11};
        RealVector      temp = new CpuDoubleVector(n);
        for (int corner = 0; corner < 4; corner++) {
            temp.gather(grid, c[corner]);
            temp.scale(w[corner]);
            out.add(temp);
        }
    }

    /** "Set at indices" idiom used throughout for BC enforcement: zero via skippedScale, then add the target value. */
    private static void setAt(RealVector field, IntegerVector idx, RealVector zeros, RealVector values) {
        field.skippedScale(zeros, idx, true, false);
        if (values != null) field.skippedAxpy(1.0, values, idx, true, false);
    }

    @Test
    void buoyantWakePastObstacle() {

        // ----------------------------------------------------------------
        // 1. Pressure Poisson, built once (static domain/obstacle): CSR
        //    rows assembled directly, same style as BentPlateVorteLaminarTest.
        //    Neumann at walls/inflow/obstacle (term dropped), Dirichlet
        //    p=0 ghost at outflow.
        // ----------------------------------------------------------------
        int[] fluidIndex = new int[nx * ny];
        Arrays.fill(fluidIndex, -1);
        List<int[]> fluidCells = new ArrayList<>();
        int nFluid = 0;
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                if (!solidCell(i, j)) {
                    fluidIndex[i * ny + j] = nFluid++;
                    fluidCells.add(new int[]{i, j});
                }

        PaddedCSRMatrix A = new PaddedCSRMatrix(nFluid, nFluid, 5);
        double[] rowVals = new double[5];
        int[] rowCols = new int[5];
        for (int k = 0; k < nFluid; k++) {
            int i = fluidCells.get(k)[0], j = fluidCells.get(k)[1];
            double diag = 0.0;
            int cnt = 0;
            int[][] nbrs = {{i - 1, j}, {i + 1, j}, {i, j - 1}, {i, j + 1}};
            for (int[] nb : nbrs) {
                int ni = nb[0], nj = nb[1];
                if (ni >= 0 && ni < nx && nj >= 0 && nj < ny) {
                    if (!solidCell(ni, nj)) {
                        rowVals[cnt] = 1.0 / (dx * dx);
                        rowCols[cnt] = fluidIndex[ni * ny + nj];
                        cnt++;
                        diag -= 1.0 / (dx * dx);
                    } // else solid neighbor: Neumann, no term
                } else if (ni >= nx) {
                    diag -= 1.0 / (dx * dx); // outflow ghost p=0
                } // left/bottom/top: Neumann, no term
            }
            rowVals[cnt] = diag; rowCols[cnt] = k; cnt++;
            A.setRow(k, Arrays.copyOf(rowVals, cnt), Arrays.copyOf(rowCols, cnt), cnt);
        }

        // ----------------------------------------------------------------
        // 2. Persistent Eulerian velocity, static BC index lists.
        // ----------------------------------------------------------------
        double[] uArr = new double[uN()];
        for (int i = 0; i <= nx; i++)
            for (int j = 0; j < ny; j++)
                uArr[uIdx(i, j)] = inflowU((j + 0.5) * dy);
        RealVector u = new CpuDoubleVector(uArr);
        RealVector v = new CpuDoubleVector(vN());

        // inflow (u): i=0, all j -- set to prescribed profile
        List<Integer> inflowList = new ArrayList<>();
        List<Double> inflowValList = new ArrayList<>();
        for (int j = 0; j < ny; j++) { inflowList.add(uIdx(0, j)); inflowValList.add(inflowU((j + 0.5) * dy)); }
        IntegerVector inflowUIdx  = toIntVec(inflowList);
        RealVector    inflowUVals = toDoubleVec(inflowValList);
        RealVector    inflowZeros = new CpuDoubleVector(inflowList.size());

        // outflow (u): i=nx target, i=nx-1 source -- zero-gradient copy
        List<Integer> outflowList = new ArrayList<>(), outflowSrcList = new ArrayList<>();
        for (int j = 0; j < ny; j++) { outflowList.add(uIdx(nx, j)); outflowSrcList.add(uIdx(nx - 1, j)); }
        IntegerVector outflowUIdx = toIntVec(outflowList);
        IntegerVector outflowSrcUIdx = toIntVec(outflowSrcList);
        RealVector    outflowZeros   = new CpuDoubleVector(outflowList.size());

        // walls (v): ground j=0 and top j=ny -- zero
        List<Integer> wallVList = new ArrayList<>();
        for (int i = 0; i < nx; i++) { wallVList.add(vIdx(i, 0)); wallVList.add(vIdx(i, ny)); }
        IntegerVector wallVIdx   = toIntVec(wallVList);
        RealVector    wallVZeros = new CpuDoubleVector(wallVList.size());

        // obstacle (u): interior faces touching a solid cell -- zero
        List<Integer> obstUList = new ArrayList<>();
        for (int i = 1; i < nx; i++)
            for (int j = 0; j < ny; j++)
                if (solidCell(i - 1, j) || solidCell(i, j)) obstUList.add(uIdx(i, j));
        IntegerVector obstUIdx   = toIntVec(obstUList);
        RealVector    obstUZeros = new CpuDoubleVector(obstUList.size());

        // obstacle (v): interior faces touching a solid cell -- zero
        List<Integer> obstVList = new ArrayList<>();
        for (int i = 0; i < nx; i++)
            for (int j = 1; j < ny; j++)
                if (solidCell(i, j - 1) || solidCell(i, j)) obstVList.add(vIdx(i, j));
        IntegerVector obstVIdx   = toIntVec(obstVList);
        RealVector    obstVZeros = new CpuDoubleVector(obstVList.size());

        applyBoundaryConditions(u, v, inflowUIdx, inflowUVals, inflowZeros,
                outflowUIdx, outflowSrcUIdx, outflowZeros,
                wallVIdx, wallVZeros, obstUIdx, obstUZeros, obstVIdx, obstVZeros, false);

        // Fixed temperature field never changes -- render it once, not per step.
        double[][] temperatureField = new double[nx][ny];
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                temperatureField[i][j] = temperatureCellCenter(i, j);
        Field2DRenderer.saveHeatmap(temperatureField, "buoyant_wake_pic/temperature.png",
                Field2DRenderer.Interpolation.NEAREST);

        // ----------------------------------------------------------------
        // 3. Particles.
        // ----------------------------------------------------------------
        ParticleSystem particles = new ParticleSystem(2,
                () -> new CpuDoubleVector(0),
                () -> new CpuBooleanVector(0));

        Random rng = new Random(0);
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++) {
                if (solidCell(i, j)) continue;
                for (int c = 0; c < perCell; c++) spawnParticle(particles,
                        (i + rng.nextDouble()) * dx, (j + rng.nextDouble()) * dy,
                        inflowU((j + 0.5) * dy), 0.0);
            }

        // ----------------------------------------------------------------
        // 4. Time loop.
        // ----------------------------------------------------------------
        WorkingBuffer<RealVector> advectScratch = new WorkingBuffer<>(6, () -> new CpuDoubleVector(0));

        for (int step = 0; step < nSteps; step++) {
            int n = particles.count();

            // --- P2G: particle velocity -> grid ---
            double[] pxArr = ((CpuDoubleVector) particles.position(0)).array();
            double[] pyArr = ((CpuDoubleVector) particles.position(1)).array();
            CornerWeights cwU = bilinearWeights(pxArr, pyArr, n, nx + 1, ny, 0.0, 0.5, BuoyantWakePicTest::uIdx);
            CornerWeights cwV = bilinearWeights(pxArr, pyArr, n, nx, ny + 1, 0.5, 0.0, BuoyantWakePicTest::vIdx);

            u = scatterToGrid(particles.velocity(0), n, uN(), cwU);
            v = scatterToGrid(particles.velocity(1), n, vN(), cwV);

            applyBoundaryConditions(u, v, inflowUIdx, inflowUVals, inflowZeros,
                    outflowUIdx, outflowSrcUIdx, outflowZeros,
                    wallVIdx, wallVZeros, obstUIdx, obstUZeros, obstVIdx, obstVZeros, false);

            // --- pressure projection ---
            double[] uRaw = ((CpuDoubleVector) u).array();
            double[] vRaw = ((CpuDoubleVector) v).array();
            double[] rhs = new double[nFluid];
            for (int k = 0; k < nFluid; k++) {
                int i = fluidCells.get(k)[0], j = fluidCells.get(k)[1];
                double div = (uRaw[uIdx(i + 1, j)] - uRaw[uIdx(i, j)]) / dx
                        + (vRaw[vIdx(i, j + 1)] - vRaw[vIdx(i, j)]) / dy;
                rhs[k] = (rho / dt) * div;
            }
            double[] pVals = BiCGStab.solve(A, rhs, 200, 1e-8);

            double[] p = new double[nx * ny];
            for (int k = 0; k < nFluid; k++) {
                int i = fluidCells.get(k)[0], j = fluidCells.get(k)[1];
                p[i * ny + j] = pVals[k];
            }
            for (int i = 1; i < nx; i++)
                for (int j = 0; j < ny; j++)
                    if (!solidCell(i - 1, j) && !solidCell(i, j))
                        uRaw[uIdx(i, j)] -= dt / rho * (p[i * ny + j] - p[(i - 1) * ny + j]) / dx;
            for (int j = 0; j < ny; j++)
                if (!solidCell(nx - 1, j))
                    uRaw[uIdx(nx, j)] -= dt / rho * (0.0 - p[(nx - 1) * ny + j]) / dx;
            for (int i = 0; i < nx; i++)
                for (int j = 1; j < ny; j++)
                    if (!solidCell(i, j - 1) && !solidCell(i, j))
                        vRaw[vIdx(i, j)] -= dt / rho * (p[i * ny + j] - p[i * ny + (j - 1)]) / dy;

            applyBoundaryConditions(u, v, inflowUIdx, inflowUVals, inflowZeros,
                    outflowUIdx, outflowSrcUIdx, outflowZeros,
                    wallVIdx, wallVZeros, obstUIdx, obstUZeros, obstVIdx, obstVZeros, false);

            // --- advect: G2P + buoyancy force, frozen grid for this outer step ---
            final RealVector uFrozen = u, vFrozen = v;
            VelocityResampler resampler = ps -> {
                int m = ps.count();
                double[] px = ((CpuDoubleVector) ps.position(0)).array();
                double[] py = ((CpuDoubleVector) ps.position(1)).array();
                CornerWeights cu = bilinearWeights(px, py, m, nx + 1, ny, 0.0, 0.5, BuoyantWakePicTest::uIdx);
                CornerWeights cv = bilinearWeights(px, py, m, nx, ny + 1, 0.5, 0.0, BuoyantWakePicTest::vIdx);
                gatherToParticles(uFrozen, ps.velocity(0), m, cu);
                gatherToParticles(vFrozen, ps.velocity(1), m, cv);

                double[] buoy = new double[m];
                for (int k = 0; k < m; k++) {
                    double t = temperatureAt(px[k], py[k]);
                    buoy[k] = g * (t - T0) / T0;   // acceleration; mass=1 below, so this doubles as force
                }
                ps.force(0).clear();
                ps.force(1).copy(new CpuDoubleVector(buoy));
            };

            if (advectScratch.get(0).size() != n) advectScratch.resize(n);
            RK2Advector.step(particles, resampler, dt, subSteps, advectScratch);

            // --- clamp to domain / push out of obstacle (raw-array escape hatch,
            //     same as the outflow-kill predicate below) --------------------
            // RK2Advector only knows about DoubleVector positions/velocities, not
            // domain geometry -- nothing stops a particle's y from drifting past
            // Ly once buoyancy has pushed it upward, since only the GRID gets a
            // wall boundary condition, never the particle itself. Left unclamped,
            // particles accumulate indefinitely along/above the top wall (worse
            // near the wake, where the redirected flow also carries them toward
            // the outflow corner) and are never culled since only x >= Lx kills.
            {
                double[] px = ((CpuDoubleVector) particles.position(0)).array();
                double[] py = ((CpuDoubleVector) particles.position(1)).array();
                double[] pu = ((CpuDoubleVector) particles.velocity(0)).array();
                double[] pv = ((CpuDoubleVector) particles.velocity(1)).array();
                for (int i = 0; i < particles.count(); i++) {
                    if (py[i] < 0) { py[i] = 0; if (pv[i] < 0) pv[i] = 0; }
                    if (py[i] > Ly) { py[i] = Ly; if (pv[i] > 0) pv[i] = 0; }

                    int ci = clampInt((int) Math.floor(px[i] / dx), 0, nx - 1);
                    int cj = clampInt((int) Math.floor(py[i] / dy), 0, ny - 1);
                    if (solidCell(ci, cj)) {
                        // Cheap push-out: snap back to just upstream of the block
                        // (same fix-up as the Python toy's push_out_of_solid).
                        px[i] = (obstacleI0 - 0.1) * dx;
                        if (pu[i] < 0) pu[i] = 0;
                    }
                }
            }

            // --- outflow cull (raw-array predicate, same escape hatch as setup code) ---
            double[] pxAfter = ((CpuDoubleVector) particles.position(0)).array();
            for (int i = 0; i < particles.count(); i++)
                if (pxAfter[i] >= Lx - 1e-6) particles.kill(i);
            WorkingBuffer<IntegerVector> compactScratch = new WorkingBuffer<>(1, () -> new CpuIntegerVector(particles.count()));
            particles.compact(compactScratch);

            // --- reseed at inflow ---
            for (int j = 0; j < ny; j++)
                for (int c = 0; c < perCell; c++)
                    spawnParticle(particles, rng.nextDouble() * 0.5 * dx, j * dy + rng.nextDouble() * dy,
                            inflowU((j + 0.5) * dy), 0.0);

            if (step % 8 == 0 || step == nSteps - 1) {
                double maxU = 0;
                double[] uf = ((CpuDoubleVector) u).array();
                for (double val : uf) maxU = Math.max(maxU, Math.abs(val));
                System.out.printf(java.util.Locale.ROOT,
                        "step=%d t=%.0fs particles=%d maxU=%.2f%n", step, step * dt, particles.count(), maxU);

                // ---- plotting, same three renderers as BentPlateVorteLaminarTest ----
                double[][] uField = new double[nx][ny];
                double[][] vField = new double[nx][ny];
                double[][] speed = new double[nx][ny];
                double[][] pressure = new double[nx][ny];
                double[][] vorticity = new double[nx][ny];
                double[][] divergence = new double[nx][ny];

                for (int i = 0; i < nx; i++)
                    for (int j = 0; j < ny; j++) {
                        if (solidCell(i, j)) continue;
                        double uc = 0.5 * (uRaw[uIdx(i, j)] + uRaw[uIdx(i + 1, j)]);
                        double vc = 0.5 * (vRaw[vIdx(i, j)] + vRaw[vIdx(i, j + 1)]);
                        uField[i][j] = uc;
                        vField[i][j] = vc;
                        speed[i][j] = Math.hypot(uc, vc);
                        pressure[i][j] = p[i * ny + j];
                        divergence[i][j] = (uRaw[uIdx(i + 1, j)] - uRaw[uIdx(i, j)]) / dx
                                + (vRaw[vIdx(i, j + 1)] - vRaw[vIdx(i, j)]) / dy;
                    }

                for (int i = 1; i < nx - 1; i++)
                    for (int j = 1; j < ny - 1; j++) {
                        if (solidCell(i, j)) continue;
                        double dvdx = (vField[i + 1][j] - vField[i - 1][j]) / (2 * dx);
                        double dudy = (uField[i][j + 1] - uField[i][j - 1]) / (2 * dy);
                        vorticity[i][j] = dvdx - dudy;
                    }

                String stepDir = "buoyant_wake_pic";

                // ---- particle mass density: sum of particle mass per cell ----
                // (post-processing/plotting code, same raw-array style as the field
                // arrays above -- not something that belongs as a DoubleVector op)
                double[][] massDensity = new double[nx][ny];
                double[] pxNow = ((CpuDoubleVector) particles.position(0)).array();
                double[] pyNow = ((CpuDoubleVector) particles.position(1)).array();
                double[] pmNow = ((CpuDoubleVector) particles.mass()).array();
                for (int k = 0; k < particles.count(); k++) {
                    int ci = clampInt((int) Math.floor(pxNow[k] / dx), 0, nx - 1);
                    int cj = clampInt((int) Math.floor(pyNow[k] / dy), 0, ny - 1);
                    massDensity[ci][cj] += pmNow[k];
                }
                Field2DRenderer.saveHeatmap(massDensity, String.format("%s/density/%05d.png", stepDir, step),
                        Field2DRenderer.Interpolation.NEAREST);
                Field2DRenderer.saveHeatmap(divergence, String.format("%s/divergence/%05d.png", stepDir, step),
                        Field2DRenderer.Interpolation.NEAREST);
                Field2DRenderer.saveHeatmap(vorticity, String.format("%s/vorticity/%05d.png", stepDir, step),
                        Field2DRenderer.Interpolation.NEAREST);
                Field2DRenderer.saveHeatmap(pressure, String.format("%s/pressure/%05d.png", stepDir, step),
                        Field2DRenderer.Interpolation.NEAREST);
                Field2DRenderer.saveHeatmap(speed, String.format("%s/speed/%05d.png", stepDir, step),
                        Field2DRenderer.Interpolation.NEAREST);
                StreamlineRenderer.saveStreamlines(uField, vField, String.format("%s/streamline/%05d.png", stepDir, step));
                VectorFieldRenderer.saveVectorField(uField, vField, String.format("%s/vect/%05d.png", stepDir, step));
            }
        }
    }

    private static void spawnParticle(ParticleSystem particles, double x, double y, double u0, double v0) {
        int idx = particles.spawn();
        setSingle(particles.position(0), idx, x);
        setSingle(particles.position(1), idx, y);
        setSingle(particles.velocity(0), idx, u0);
        setSingle(particles.velocity(1), idx, v0);
        setSingle(particles.mass(), idx, 1.0);
    }

    /**
     * Single-index "set" on a DoubleVector, expressed with the skippedAxpy
     * idiom (baseline is guaranteed 0 from resize-on-grow, so += equals =).
     * Fine for one-off spawn-time initialization; not meant for hot loops.
     */
    private static void setSingle(RealVector field, int idx, double value) {
        IntegerVector one = new CpuIntegerVector(1);
        one.set(idx, 0);
        RealVector val = new CpuDoubleVector(new double[]{value});
        field.skippedAxpy(1.0, val, one, true, false);
    }

    /**
     * Enforces inflow / wall / obstacle every call. {@code applyOutflow}
     * controls whether the outflow zero-gradient copy runs.
     *
     * <p>Must be {@code false} for the pre-solve call: forcing
     * {@code u[nx,:] := u[nx-1,:]} before the divergence RHS is computed
     * makes that column's east-face divergence term exactly zero by
     * construction (not because the flow is actually divergence-free
     * there) -- which contradicts the Dirichlet-p=0 ghost outflow anchor
     * baked into the pressure matrix, and the solver "resolves" that
     * contradiction by pushing pressure (and velocity) to extremes at
     * that boundary, worse each step. The *only* place u[nx,:] should be
     * set is the post-solve ghost-pressure correction.
     */
    private static void applyBoundaryConditions(RealVector u, RealVector v,
                                                IntegerVector inflowUIdx, RealVector inflowUVals, RealVector inflowZeros,
                                                IntegerVector outflowUIdx, IntegerVector outflowSrcUIdx, RealVector outflowZeros,
                                                IntegerVector wallVIdx, RealVector wallVZeros,
                                                IntegerVector obstUIdx, RealVector obstUZeros,
                                                IntegerVector obstVIdx, RealVector obstVZeros,
                                                boolean applyOutflow) {
        setAt(u, inflowUIdx, inflowZeros, inflowUVals);
        if (applyOutflow) {
            RealVector outflowSrc = new CpuDoubleVector(outflowSrcUIdx.size());
            outflowSrc.gather(u, outflowSrcUIdx);
            setAt(u, outflowUIdx, outflowZeros, outflowSrc);
        }
        setAt(v, wallVIdx, wallVZeros, null);
        setAt(u, obstUIdx, obstUZeros, null);
        setAt(v, obstVIdx, obstVZeros, null);
    }

    private static IntegerVector toIntVec(List<Integer> list) {
        IntegerVector v = new CpuIntegerVector(list.size());
        for (int i = 0; i < list.size(); i++) v.set(list.get(i), i);
        return v;
    }

    private static RealVector toDoubleVec(List<Double> list) {
        double[] arr = new double[list.size()];
        for (int i = 0; i < list.size(); i++) arr[i] = list.get(i);
        return new CpuDoubleVector(arr);
    }
}