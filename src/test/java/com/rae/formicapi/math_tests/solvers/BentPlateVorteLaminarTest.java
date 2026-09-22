package com.rae.formicapi.math_tests.solvers;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR3Tensor;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import com.rae.formicapi.foundation.math.operators.vectors.WorkingBuffer;
import com.rae.formicapi.foundation.math.solvers.NewtonKrylov;
import com.rae.formicapi.foundation.plotting.StreamlineRenderer;
import com.rae.formicapi.foundation.plotting.VectorFieldRenderer;
import org.junit.jupiter.api.Test;

/**
 * Plain laminar incompressible flow past a flat plate held normal
 * (broadside) to the freestream -- same architecture as
 * {@code CylinderVortexSheddingLaminarTest} (state x=[u,v,p], same
 * inlet/outlet/far-field treatment, same local-viscosity-blend trick near
 * the body), just a rectangular immersed obstacle instead of a circle.
 *
 * <p>Chosen specifically to see clean periodic laminar eddies: unlike a
 * circular cylinder, a normal flat plate has its separation points pinned
 * at the sharp edges rather than depending on where the boundary layer
 * happens to detach, so shedding onset happens at a much lower Re (order
 * 10-30 in the literature, vs ~47 for a cylinder) and the wake tends to be
 * more regular. Expect the corners to have sharper local gradients than the
 * cylinder's smooth surface did, though -- the viscosity-blend parameters
 * below are carried over from the cylinder test as a starting point, not
 * independently re-tuned for this geometry.
 *
 * <p>Re_H = U_inf*H/nu (H = frontal height, the flat-plate analogue of
 * diameter) with the same far/near split as before. Expected Strouhal for a
 * normal flat plate is commonly cited around St~0.145 -- a rougher
 * literature number than Williamson's fitted cylinder correlation, not a
 * precise formula, so treat the printed expected period as an order-of-
 * magnitude check, not a target to match exactly.
 */
public class BentPlateVorteLaminarTest {

    enum DofRole {
        INTERIOR,
        DIRICHLET, NEUMANN,
        RADIATION,   // reserved: future PML absorbing-layer boundary, not implemented yet
        FREE_OUTFLOW
    }

    private static double plateCenterlineX(double py, double plateCenterX, double plateCenterY,
                                           double plateHalfHeight, double bendAmount) {
        double s = (py - plateCenterY) / plateHalfHeight;
        s = Math.max(-1.0, Math.min(1.0, s));
        return plateCenterX + bendAmount * s * s;
    }

    private static double distanceToBentPlate(int x, int y, double hx, double hy,
                                              double plateCenterX, double plateCenterY,
                                              double plateHalfThickness, double plateHalfHeight,
                                              double bendAmount) {
        double px = x * hx, py = y * hy;

        if (py < plateCenterY - plateHalfHeight || py > plateCenterY + plateHalfHeight)
            return Math.abs(px - plateCenterX);

        double cx = plateCenterlineX(py, plateCenterX, plateCenterY,
                plateHalfHeight, bendAmount);

        return Math.max(0.0, Math.abs(px - cx) - plateHalfThickness);
    }

    /**
     * {@code nu} varying smoothly from {@code nuNear} at the plate's surface
     * down to {@code nuFar} beyond {@code blendWidth} -- see the class-level
     * note on why this only needs to stabilize the region with actually
     * sharp gradients, not the whole domain. Distance here is to the nearest
     * point of the plate's rectangle (0 inside/on it, standard "distance to
     * an axis-aligned box" outside), not a circle's radius.
     */
    private static double localViscosity(int x, int y, double hx, double hy,
                                         double plateCenterX, double plateCenterY,
                                         double plateHalfThickness, double plateHalfHeight,
                                         double bendAmount,
                                         double nuFar, double nuNear, double blendWidth) {

        double dSurf = distanceToBentPlate(x, y, hx, hy,
                plateCenterX, plateCenterY,
                plateHalfThickness, plateHalfHeight,
                bendAmount);

        double t = Math.max(0.0, Math.min(1.0, dSurf / blendWidth));
        double smooth = 1.0 - t * t * (3.0 - 2.0 * t);

        return nuFar + (nuNear - nuFar) * smooth;
    }

    static void buildPlateOperator(int nx, int ny, double hx, double hy, double nuFar, double nuNear, double blendWidth,
                                   double plateCenterX, double plateCenterY, double plateHalfThickness, double plateHalfHeight,
                                   double bendAmount,
                                   double uInf, double pressurePenalty, double invDt,
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

                double px = x * hx, py = y * hy;

                scaling[u] = 1;
                scaling[v] = 1;
                scaling[p] = 1 / 10.0;

                boolean isInlet    = x == 0;
                boolean isOutlet   = !isInlet && x == nx - 1;
                boolean isFarField = !isInlet && !isOutlet && (y == 0 || y == ny - 1);
                double plateX = plateCenterlineX(py, plateCenterX, plateCenterY,
                        plateHalfHeight, bendAmount);

                boolean isPlate = !isInlet && !isOutlet && !isFarField
                        && Math.abs(py - plateCenterY) <= plateHalfHeight
                        && Math.abs(px - plateX) <= plateHalfThickness;

                if (isInlet) {
                    A.setRow(u, new double[]{1}, new int[]{u}, 1);
                    A.setRow(v, new double[]{1}, new int[]{v}, 1);
                    role[u] = DofRole.DIRICHLET; bcValue[u] = uInf;
                    role[v] = DofRole.DIRICHLET; bcValue[v] = 0;

                    int east = u + 3;
                    A.setRow(p, new double[]{1, -1}, new int[]{p, east + 2}, 2);
                    role[p] = DofRole.NEUMANN;

                    C.setRow(u, new double[0], new int[0], new int[0], 0);
                    C.setRow(v, new double[0], new int[0], new int[0], 0);
                    C.setRow(p, new double[0], new int[0], new int[0], 0);
                    continue;
                }

                if (isOutlet) {
                    // Zero-curvature ("do-nothing") outflow -- see the
                    // cylinder test for why this is preferred over plain
                    // zero-gradient at a genuine advective outlet.
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
                    // wall -- see the cylinder test for the full reasoning.
                    A.setRow(u, new double[]{1}, new int[]{u}, 1);
                    A.setRow(v, new double[]{1}, new int[]{v}, 1);
                    role[u] = DofRole.DIRICHLET; bcValue[u] = uInf;
                    role[v] = DofRole.DIRICHLET; bcValue[v] = 0;

                    int pNeighbor = (y == 0) ? p + 3 * nx : p - 3 * nx;
                    A.setRow(p, new double[]{1, -1}, new int[]{p, pNeighbor}, 2);
                    role[p] = DofRole.NEUMANN;

                    C.setRow(u, new double[0], new int[0], new int[0], 0);
                    C.setRow(v, new double[0], new int[0], new int[0], 0);
                    C.setRow(p, new double[0], new int[0], new int[0], 0);
                    continue;
                }

                if (isPlate) {
                    A.setRow(u, new double[]{1}, new int[]{u}, 1);
                    A.setRow(v, new double[]{1}, new int[]{v}, 1);
                    role[u] = DofRole.DIRICHLET; bcValue[u] = 0;
                    role[v] = DofRole.DIRICHLET; bcValue[v] = 0;

                    // Same harmonic-extension trick as the cylinder's solid
                    // interior: 4*p_self - (p_e+p_w+p_n+p_s) = 0, uniformly,
                    // regardless of which neighbors happen to be fluid --
                    // see the cylinder test for the full rationale. Shape-
                    // agnostic, so it transfers unchanged to a rectangle.
                    //A.setRow(p, new double[]{4, -1, -1, -1, -1}, new int[]{p, p + 3, p - 3, p + 3 * nx, p - 3 * nx}, 5);
                    A.setRow(p, new double[]{1}, new int[]{p}, 1);
                    role[p] = DofRole.NEUMANN;

                    C.setRow(u, new double[0], new int[0], new int[0], 0);
                    C.setRow(v, new double[0], new int[0], new int[0], 0);
                    C.setRow(p, new double[0], new int[0], new int[0], 0);
                    continue;
                }

                // ---------------- fluid cell: plain laminar Navier-Stokes ----------------

                role[u] = DofRole.INTERIOR;
                role[v] = DofRole.INTERIOR;
                role[p] = DofRole.DIRICHLET;
                bcValue[p] = 0;

                int west = 3 * (cell - 1), east = 3 * (cell + 1);
                int south = 3 * (cell - nx), north = 3 * (cell + nx);

                boolean solidW = isPlateCell(x - 1, y, hx, hy, plateCenterX, plateCenterY,
                        plateHalfThickness, plateHalfHeight, bendAmount);

                boolean solidE = isPlateCell(x + 1, y, hx, hy, plateCenterX, plateCenterY,
                        plateHalfThickness, plateHalfHeight, bendAmount);

                boolean solidS = isPlateCell(x, y - 1, hx, hy, plateCenterX, plateCenterY,
                        plateHalfThickness, plateHalfHeight, bendAmount);

                boolean solidN = isPlateCell(x, y + 1, hx, hy, plateCenterX, plateCenterY,
                        plateHalfThickness, plateHalfHeight, bendAmount);

                double nuSelf  = localViscosity(x,     y,     hx, hy, plateCenterX, plateCenterY, plateHalfThickness, plateHalfHeight, bendAmount,nuFar, nuNear, blendWidth);
                double nuEast  = localViscosity(x + 1, y,     hx, hy, plateCenterX, plateCenterY, plateHalfThickness, plateHalfHeight, bendAmount, nuFar, nuNear, blendWidth);
                double nuWest  = localViscosity(x - 1, y,     hx, hy, plateCenterX, plateCenterY, plateHalfThickness, plateHalfHeight, bendAmount,nuFar, nuNear, blendWidth);
                double nuNorth = localViscosity(x,     y + 1, hx, hy, plateCenterX, plateCenterY, plateHalfThickness, plateHalfHeight, bendAmount,nuFar, nuNear, blendWidth);
                double nuSouth = localViscosity(x,     y - 1, hx, hy, plateCenterX, plateCenterY, plateHalfThickness, plateHalfHeight, bendAmount,nuFar, nuNear, blendWidth);

                double nuFaceE = 2 * nuSelf * nuEast  / (nuSelf + nuEast);
                double nuFaceW = 2 * nuSelf * nuWest  / (nuSelf + nuWest);
                double nuFaceN = 2 * nuSelf * nuNorth / (nuSelf + nuNorth);
                double nuFaceS = 2 * nuSelf * nuSouth / (nuSelf + nuSouth);

                uRow[0] = (nuFaceE + nuFaceW) * invHx2 + (nuFaceN + nuFaceS) * invHy2 + invDt;
                uRow[1] = -nuFaceW * invHx2; uRow[2] = -nuFaceE * invHx2;
                uRow[3] = -nuFaceS * invHy2; uRow[4] = -nuFaceN * invHy2;
                uRow[5] = 0.5 * invHx; uRow[6] = -0.5 * invHx;
                uCols[0] = u; uCols[1] = west; uCols[2] = east; uCols[3] = south; uCols[4] = north;
                uCols[5] = east + 2; uCols[6] = west + 2;
                A.setRow(u, uRow, uCols, 7);

                vRow[0] = (nuFaceE + nuFaceW) * invHx2 + (nuFaceN + nuFaceS) * invHy2 + invDt;
                vRow[1] = -nuFaceW * invHx2; vRow[2] = -nuFaceE * invHx2;
                vRow[3] = -nuFaceS * invHy2; vRow[4] = -nuFaceN * invHy2;
                vRow[5] = 0.5 * invHy; vRow[6] = -0.5 * invHy;
                vCols[0] = v; vCols[1] = west + 1; vCols[2] = east + 1; vCols[3] = south + 1; vCols[4] = north + 1;
                vCols[5] = north + 2; vCols[6] = south + 2;
                A.setRow(v, vRow, vCols, 7);

                pRow[0] = 0.5 * invHx; pRow[1] = -0.5 * invHx;
                pRow[2] = 0.5 * invHy; pRow[3] = -0.5 * invHy;
                pRow[4] = pressurePenalty;
                pCols[0] = east; pCols[1] = west; pCols[2] = north + 1; pCols[3] = south + 1; pCols[4] = p;
                A.setRow(p, pRow, pCols, 5);

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

    private static boolean isPlateCell(int x, int y, double hx, double hy,
                                       double plateCenterX, double plateCenterY,
                                       double plateHalfThickness, double plateHalfHeight,
                                       double bendAmount) {
        double px = x * hx;
        double py = y * hy;

        double plateX = plateCenterlineX(py, plateCenterX, plateCenterY,
                plateHalfHeight, bendAmount);

        return Math.abs(py - plateCenterY) <= plateHalfHeight
                && Math.abs(px - plateX) <= plateHalfThickness;
    }

    @Test
    void laminarEddiesBehindFlatPlate() {

        int nx = 350;
        int ny = 280;

        double hx = 1;
        double hy = 1;

        double H = 100.0;                 // frontal height presented to the flow (flat-plate analogue of D)
        double halfHeight = H / 2;
        double thickness = 2.0;          // thin plate, 2 cells in the flow direction
        double halfThickness = thickness / 2.0;
        double plateCenterX = 80;       //2.0 * H;
        double plateCenterY = (ny - 1) * hy / 2.0;

        double bendAmount = 60.0;

        double uInf = 1.0;
        double nuFar = 0.5;                          // nominal far-field/wake viscosity -- Re_H=400 there
        double nuNear = 1;//uInf * hx * 10.0;              // Pe=5 at the surface: 0.2 (carried over from the cylinder test)
        double blendWidth = 4.0 * halfThickness;
        double ReFar = uInf * H / nuFar;
        double ReNear = uInf * H / nuNear;

        double pressurePenalty = 1e-2;

        double dt = 0.1;

        System.out.printf("Re_H_far=%.0f, Re_H_near=%.0f (blend width %.1f)", ReFar, ReNear, blendWidth);

        int nSteps = 1000;
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

        buildPlateOperator(nx, ny, hx, hy, nuFar, nuNear, blendWidth,
                plateCenterX, plateCenterY, halfThickness, halfHeight, bendAmount, uInf,
                pressurePenalty, 1.0 / dt, C, A, scaling, role, bcValue);

        // Initial condition: uniform freestream. Unlike the cylinder, a
        // normal flat plate's separation points are pinned at the sharp
        // edges, so shedding tends to start on its own without needing a
        // deliberate symmetry-breaking perturbation -- left disabled here,
        // same as the cylinder test currently has it; uncomment if shedding
        // doesn't establish on its own.
        double bumpX = plateCenterX + 1.5 * H;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {

                int cell = j * nx + i;
                int u = 3 * cell, v = u + 1;

                if (role[u] == DofRole.INTERIOR) x[u] = uInf;
                else if (role[u] == DofRole.DIRICHLET) x[u] = bcValue[u];

                if (role[v] == DofRole.INTERIOR) {
                    double px = i * hx, py = j * hy;
                    x[v] = 0;//0.02 * uInf * Math.exp(-Math.pow((px - bumpX) / H, 2)) * ((py - plateCenterY) / halfHeight);
                } else if (role[v] == DofRole.DIRICHLET) x[v] = bcValue[v];
            }
        }

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

            if (step % printEvery == 0 || step == nSteps - 1) {

                int linearIt = 0;
                for (int i: stats.linearIterations){
                    linearIt += i;
                }

                System.out.printf(java.util.Locale.ROOT,
                        "t=%.2f step=%d/%d newtonIters=%d linearIters=%d converged=%b %n",
                        t, step, nSteps, stats.newtonIterationCount, linearIt,
                        stats.converged);

                double[][] uField = new double[nx][ny];
                double[][] vField = new double[nx][ny];
                double[][] speed = new double[nx][ny];
                double[][] vorticity = new double[nx][ny];
                double[][] pressure = new double[nx][ny];
                double[][] divergence = new double[nx][ny];

                for (int j = 1; j < ny - 1; j++) {
                    for (int i = 1; i < nx - 1; i++) {
                        int cell = j * nx + i;
                        int east = 3 * (cell + 1);
                        int west = 3 * (cell - 1);
                        int north = 3 * (cell + nx);
                        int south = 3 * (cell - nx);

                        divergence[i][j] =
                                (x[east] - x[west]) * 0.5 * invHx
                                        + (x[north + 1] - x[south + 1]) * 0.5 * invHy;
                    }
                }


                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++) {
                        int cell = j * nx + i;
                        int u = 3 * cell, v = u + 1, p = v + 1;
                        uField[i][j] = x[u];
                        vField[i][j] = x[v];
                        speed[i][j] = Math.hypot(x[u], x[v]);
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

                com.rae.formicapi.foundation.plotting.Field2DRenderer.saveHeatmap(divergence, String.format("bent_plate_laminar/divergence/%05d.png", step),
                        com.rae.formicapi.foundation.plotting.Field2DRenderer.Interpolation.NEAREST);
                com.rae.formicapi.foundation.plotting.Field2DRenderer.saveHeatmap(vorticity, String.format("bent_plate_laminar/vorticity/%05d.png", step),
                        com.rae.formicapi.foundation.plotting.Field2DRenderer.Interpolation.NEAREST);
                com.rae.formicapi.foundation.plotting.Field2DRenderer.saveHeatmap(pressure, String.format("bent_plate_laminar/pressure/%05d.png", step),
                        com.rae.formicapi.foundation.plotting.Field2DRenderer.Interpolation.NEAREST);
                StreamlineRenderer.saveStreamlines(uField, vField,
                        String.format("bent_plate_laminar/streamline/%05d.png", step));
                VectorFieldRenderer.saveVectorField(uField, vField,
                        String.format("bent_plate_laminar/vect/%05d.png", step));
            }
        }
    }
}