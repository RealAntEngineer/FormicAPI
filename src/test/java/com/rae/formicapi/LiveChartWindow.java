package com.rae.formicapi;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;

import java.awt.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Wraps an XChart {@link XYChart} for live-updating display during a long
 * run, without blowing up in headless environments (CI runners, servers
 * with no X display, etc).
 *
 * <p>When a display is available, this opens an interactive Swing window
 * and {@link #repaint()} refreshes it. When headless, {@link #repaint()}
 * is a no-op and {@link #saveSnapshot(String)} writes the chart to a PNG
 * instead, so tests still produce a checkable artifact instead of throwing
 * {@code HeadlessException}.
 */
public final class LiveChartWindow {

    private final XYChart               chart;
    private final SwingWrapper<XYChart> window;
    private final boolean interactive;

    public LiveChartWindow(XYChart chart) {
        this.chart = chart;
        this.interactive = !GraphicsEnvironment.isHeadless();

        if (interactive) {
            this.window = new SwingWrapper<>(chart);
            this.window.displayChart();
        } else {
            this.window = null;
        }
    }

    /** Refreshes the live window after the chart's series data has been updated. No-op when headless. */
    public void repaint() {
        if (interactive)
            window.repaintChart();
    }

    public boolean isInteractive() {
        return interactive;
    }

    /** Writes the current chart state to test-output/{filename}.png -- useful as a periodic checkpoint or CI artifact. */
    public void saveSnapshot(String filename) {
        try {
            Path path = Paths.get("test-output", filename);
            Files.createDirectories(path.getParent());
            BitmapEncoder.saveBitmap(chart, path.toString(), BitmapEncoder.BitmapFormat.PNG);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
