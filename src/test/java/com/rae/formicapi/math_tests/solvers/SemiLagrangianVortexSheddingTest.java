package com.rae.formicapi.math_tests.solvers;

import com.rae.formicapi.foundation.plotting.LiveChartWindow;
import com.rae.formicapi.foundation.plotting.StreamlineRenderer;
import com.rae.formicapi.foundation.plotting.VectorFieldRenderer;
import org.junit.jupiter.api.Test;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.Styler;

import java.util.ArrayList;
import java.util.List;

import static com.google.common.primitives.Doubles.toArray;

/**
 * Wind-tunnel scene for {@link StableFluidGrid} -- direct adaptation of the
 * "vortex shedding" scene from the Ten Minute Physics reference this was
 * ported from: inlet velocity at the left, solid top/bottom walls, a
 * circular obstacle, and a smoke ribbon injected at the inlet to visualize
 * mixing in the wake. See {@link StableFluidGrid}'s class javadoc for how
 * this compares to the Newton-Krylov-based cylinder/plate tests in this
 * package.
 */
public class SemiLagrangianVortexSheddingTest {

    @Test
    void windTunnelAroundCylinder() {

        double domainHeight = 1.0;
        double aspect = 3.0;                 // wide tunnel, room for the wake to develop
        double domainWidth = aspect * domainHeight;

        int res = 150;                       // cells per unit height
        double h = domainHeight / res;
        int numX = (int) Math.round(domainWidth / h);
        int numY = res;

        double density = 1000.0;
        double inVel = 2.0;
        double gravity = 0.0;
        double overRelaxation = 1.9;          // SOR factor, matches the reference
        int numIters = 60;                    // Gauss-Seidel sweeps per step
        double dt = 1.0 / 60.0;

        double obstacleRadius = 0.15 * domainHeight;
        double obstacleX = 0.25 * domainWidth;
        double obstacleY = 0.5 * domainHeight;

        int nSteps = 2000;
        int printEvery = 20;
        int sampleEvery = 2;

        StableFluidGrid grid = new StableFluidGrid(density, numX, numY, h);
        int n = grid.numY;

        // Left wall (with inlet velocity punched through it) + top/bottom
        // walls solid; right edge intentionally left open -- solveIncompressibility
        // never touches the boundary columns/rows, so the outlet is "open"
        // by omission (advection just carries things out), not an explicit
        // BC the way the finite-difference tests do it. Matches the
        // reference's own tested behavior.
        for (int i = 0; i < grid.numX; i++) {
            for (int j = 0; j < grid.numY; j++) {
                double sVal = 1.0;
                if (i == 0 || j == 0 || j == grid.numY - 1) sVal = 0.0;
                grid.s[i * n + j] = sVal;
                if (i == 1) grid.u[i * n + j] = inVel;
            }
        }

        // Smoke ribbon at the inlet (m=0 band on an m=1 background), same
        // proportion as the reference (10% of the channel height).
        double pipeH = 0.1 * grid.numY;
        int minJ = (int) Math.floor(0.5 * grid.numY - 0.5 * pipeH);
        int maxJ = (int) Math.floor(0.5 * grid.numY + 0.5 * pipeH);
        for (int j = minJ; j < maxJ; j++) grid.m[j] = 0.0;

        // Static circular obstacle.
        for (int i = 1; i < grid.numX - 2; i++) {
            for (int j = 1; j < grid.numY - 2; j++) {
                grid.s[i * n + j] = 1.0;
                double dx = (i + 0.5) * h - obstacleX;
                double dy = (j + 0.5) * h - obstacleY;
                if (dx * dx + dy * dy < obstacleRadius * obstacleRadius) {
                    grid.s[i * n + j] = 0.0;
                    grid.m[i * n + j] = 1.0;
                    grid.u[i * n + j] = 0.0; grid.u[(i + 1) * n + j] = 0.0;
                    grid.v[i * n + j] = 0.0; grid.v[i * n + j + 1] = 0.0;
                }
            }
        }

        // Wake probe: 4 diameters downstream of the obstacle, centerline height.
        int probeI = (int) Math.round((obstacleX + 8 * obstacleRadius) / h);
        int probeJ = (int) Math.round(obstacleY / h);

        XYChart wakeChart = new XYChartBuilder()
                .width(900).height(400)
                .title("Wake probe v-velocity (cell-centered, 4D downstream)")
                .xAxisTitle("t").yAxisTitle("v").build();
        wakeChart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);
        wakeChart.addSeries("v(probe)", new double[]{0}, new double[]{0});
        LiveChartWindow wakeWindow = new LiveChartWindow(wakeChart);

        List<Double> tSampled = new ArrayList<>();
        List<Double> vSampled = new ArrayList<>();

        for (int step = 0; step < nSteps; step++) {

            grid.simulate(dt, gravity, numIters, overRelaxation);

            double t = step * dt;
            double vProbe = 0.5 * (grid.v[probeI * n + probeJ] + grid.v[probeI * n + probeJ + 1]);

            if (step % sampleEvery == 0 || step == nSteps - 1) {
                tSampled.add(t);
                vSampled.add(vProbe);
            }

            if (step % printEvery == 0 || step == nSteps - 1) {

                wakeChart.updateXYSeries("v(probe)", toArray(tSampled), toArray(vSampled), null);
                wakeWindow.repaint();
                if (!wakeWindow.isInteractive())
                    wakeWindow.saveSnapshot("semi_lagrangian_latest.png");

                System.out.printf(java.util.Locale.ROOT,
                        "t=%.2f step=%d/%d maxDiv=%.4g v(probe)=%.4f%n",
                        t, step, nSteps, grid.maxDivergence(), vProbe);

                double[][] uField = new double[grid.numX][grid.numY];
                double[][] vField = new double[grid.numX][grid.numY];
                double[][] speed = new double[grid.numX][grid.numY];
                double[][] vorticity = new double[grid.numX][grid.numY];
                double[][] pressure = new double[grid.numX][grid.numY];
                double[][] smoke = new double[grid.numX][grid.numY];

                // Cell-centered velocity, averaged from the staggered faces
                // -- same convention advectSmoke() uses internally.
                for (int i = 0; i < grid.numX; i++) {
                    for (int j = 0; j < grid.numY; j++) {
                        int uE = Math.min(i + 1, grid.numX - 1);
                        int vN = Math.min(j + 1, grid.numY - 1);
                        double uc = 0.5 * (grid.u[i * n + j] + grid.u[uE * n + j]);
                        double vc = 0.5 * (grid.v[i * n + j] + grid.v[i * n + vN]);
                        uField[i][j] = uc;
                        vField[i][j] = vc;
                        speed[i][j] = Math.hypot(uc, vc);
                        pressure[i][j] = grid.p[i * n + j];
                        smoke[i][j] = grid.m[i * n + j];
                    }
                }

                for (int i = 1; i < grid.numX - 1; i++) {
                    for (int j = 1; j < grid.numY - 1; j++) {
                        double dvdx = (vField[i + 1][j] - vField[i - 1][j]) * 0.5 / h;
                        double dudy = (uField[i][j + 1] - uField[i][j - 1]) * 0.5 / h;
                        vorticity[i][j] = dvdx - dudy;
                    }
                }

                com.rae.formicapi.foundation.plotting.Field2DRenderer.saveHeatmap(speed, String.format("semi_lagrangian/speed/%05d.png", step),
                        com.rae.formicapi.foundation.plotting.Field2DRenderer.Interpolation.NEAREST);
                com.rae.formicapi.foundation.plotting.Field2DRenderer.saveHeatmap(vorticity, String.format("semi_lagrangian/vorticity/%05d.png", step),
                        com.rae.formicapi.foundation.plotting.Field2DRenderer.Interpolation.NEAREST);
                com.rae.formicapi.foundation.plotting.Field2DRenderer.saveHeatmap(pressure, String.format("semi_lagrangian/pressure/%05d.png", step),
                        com.rae.formicapi.foundation.plotting.Field2DRenderer.Interpolation.NEAREST);
                com.rae.formicapi.foundation.plotting.Field2DRenderer.saveHeatmap(smoke, String.format("semi_lagrangian/smoke/%05d.png", step),
                        com.rae.formicapi.foundation.plotting.Field2DRenderer.Interpolation.BILINEAR);
                StreamlineRenderer.saveStreamlines(uField, vField,
                        String.format("semi_lagrangian/streamline/%05d.png", step));
                VectorFieldRenderer.saveVectorField(uField, vField,
                        String.format("semi_lagrangian/vect/%05d.png", step));
            }
        }
    }
}