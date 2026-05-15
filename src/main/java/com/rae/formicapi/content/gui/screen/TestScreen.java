package com.rae.formicapi.content.gui.screen;

import com.rae.formicapi.content.gui.elements.simulation.NodeWidget;
import com.rae.formicapi.content.gui.elements.simulation.SimulationComponentWidget;
import com.rae.formicapi.fondation.simulation.nodal.ModelType;
import com.rae.formicapi.fondation.simulation.nodal.SteadyStateSolver;
import com.rae.formicapi.fondation.simulation.nodal.core.*;
import net.createmod.catnip.gui.AbstractSimiScreen;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.components.EditBox;
import net.minecraft.network.chat.Component;
import org.lwjgl.system.NonnullDefault;

@NonnullDefault
public class TestScreen extends AbstractSimiScreen {

    private SimulationModel model;
    private UnknownNode    nodeA;
    private FixedValueNode nodeB;
    private LinearLink thermalLink;
    private Source     heatSource;

    private NodeWidget widgetA;
    private NodeWidget                widgetB;
    private SimulationComponentWidget linkWidget;
    private SimulationComponentWidget sourceWidget;

    private Button  solveButton;
    private Button  resetButton;
    private EditBox sourceValueBox;
    private EditBox conductanceValueBox;

    private boolean isSolved = false;
    private static final int NODE_SPACING = 200;

    @Override
    protected void init() {
        super.init();

        // Initialize the simulation model
        initializeModel();

        // Calculate center positions for nodes
        int centerX = this.width / 2;
        int centerY = this.height / 2;

        // Add node widgets
        widgetA = new NodeWidget(nodeA, centerX - NODE_SPACING / 2, centerY);
        widgetB = new NodeWidget(nodeB, centerX + NODE_SPACING / 2, centerY);

        this.addRenderableWidget(widgetA);
        this.addRenderableWidget(widgetB);

        // Add component widgets (visual representations of links/sources)
        /*linkWidget = new SimulationComponentWidget(thermalLink,
                centerX - NODE_SPACING / 2,
                centerY,
                centerX + NODE_SPACING / 2,
                centerY);*/
        sourceWidget = new SimulationComponentWidget(heatSource,
                centerX - NODE_SPACING / 2 - 50,
                centerY - 60);

        //this.addRenderableWidget(linkWidget);
        this.addRenderableWidget(sourceWidget);

        // Add control buttons
        solveButton = Button.builder(Component.literal("Solve Simulation"),
                        button -> solveSteadyState())
                .bounds(this.width / 2 - 100, this.height - 60, 90, 20)
                .build();

        resetButton = Button.builder(Component.literal("Reset"),
                        button -> resetSimulation())
                .bounds(this.width / 2 + 10, this.height - 60, 90, 20)
                .build();

        this.addRenderableWidget(solveButton);
        this.addRenderableWidget(resetButton);

        // Add input fields for parameters
        sourceValueBox = new EditBox(this.font,
                20, this.height - 100,
                100, 20,
                Component.literal("Source Power"));
        sourceValueBox.setValue("100");
        sourceValueBox.setMaxLength(10);

        conductanceValueBox = new EditBox(this.font,
                20, this.height - 70,
                100, 20,
                Component.literal("Conductance"));
        conductanceValueBox.setValue("10");
        conductanceValueBox.setMaxLength(10);

        this.addRenderableWidget(sourceValueBox);
        this.addRenderableWidget(conductanceValueBox);

        // Add labels
        addLabel("Heat Source (W):", 20, this.height - 115);
        addLabel("Conductance (W/K):", 20, this.height - 85);
    }

    private void initializeModel() {
        model = new SimulationModel();

        // Create nodes
        nodeA = new UnknownNode(ModelType.THERMAL);

        nodeB = new FixedValueNode(ModelType.THERMAL, 0);

        model.addNode(nodeA);
        model.addNode(nodeB);

        // Create components
        thermalLink = new LinearLink(nodeA, nodeB, ModelType.THERMAL, 10);
        heatSource = new Source(nodeA, ModelType.THERMAL, 100);

        model.addComponent(thermalLink);
        model.addComponent(heatSource);
    }

    private void solveSteadyState() {
        try {
            // Update parameters from input boxes
            double sourceValue = Double.parseDouble(sourceValueBox.getValue());
            double conductanceValue = Double.parseDouble(conductanceValueBox.getValue());

            // Recreate model with new parameters
            model = new SimulationModel();

            nodeA = new UnknownNode(ModelType.THERMAL);
            nodeB = new FixedValueNode(ModelType.THERMAL, 0);

            model.addNode(nodeA);
            model.addNode(nodeB);

            thermalLink = new LinearLink(nodeA, nodeB, ModelType.THERMAL, conductanceValue);
            heatSource = new Source(nodeA, ModelType.THERMAL, sourceValue);

            model.addComponent(thermalLink);
            model.addComponent(heatSource);

            // Update widgets to reference new nodes
            widgetA.node = nodeA;
            widgetB.node = nodeB;

            // Solve the system
            SteadyStateSolver.solve(model);

            isSolved = true;

        } catch (NumberFormatException e) {
            // Handle invalid input
            isSolved = false;
        }
    }

    private void resetSimulation() {
        initializeModel();
        widgetA.node = nodeA;
        widgetB.node = nodeB;
        sourceValueBox.setValue("100");
        conductanceValueBox.setValue("10");
        isSolved = false;
    }

    private void addLabel(String text, int x, int y) {
        // You can add text labels as non-interactive widgets
        this.addRenderableOnly((graphics, mouseX, mouseY, partialTicks) -> {
            graphics.drawString(this.font, text, x, y, 0xFFFFFFFF);
        });
    }

    @Override
    protected void renderWindow(GuiGraphics graphics, int mouseX, int mouseY, float partialTicks) {
        /*// Render background
        graphics.fill(0, 0, this.width, this.height, 0x90000000);

        // Render title
        Component title = Component.literal("Two Node Thermal Conduction Test");
        graphics.drawCenteredString(this.font, title, this.width / 2, 20, 0xFFFFFFFF);

        // Render simulation info panel
        renderInfoPanel(graphics);

        // Render connection line between nodes
        if (widgetA != null && widgetB != null) {
            renderConnectionLine(graphics, widgetA, widgetB);
        }

        // Render heat source visualization
        if (isSolved) {
            renderHeatFlow(graphics);
        }

        // Render results if solved
        if (isSolved) {
            renderResults(graphics);
        }*/
    }

    private void renderInfoPanel(GuiGraphics graphics) {
        int panelX = this.width - 220;
        int panelY = 40;
        int panelWidth = 200;
        int panelHeight = 120;

        // Panel background
        graphics.fill(panelX, panelY, panelX + panelWidth, panelY + panelHeight, 0xCC000000);
        graphics.renderOutline(panelX, panelY, panelWidth, panelHeight, 0xFF4A90E2);

        // Panel title
        graphics.drawString(this.font, "Simulation Info", panelX + 10, panelY + 10, 0xFFFFD700);

        // Status
        String status = isSolved ? "Solved" : "Not Solved";
        int statusColor = isSolved ? 0xFF00FF00 : 0xFFFF4500;
        graphics.drawString(this.font, "Status: " + status, panelX + 10, panelY + 30, statusColor);

        if (isSolved) {
            double tempA = nodeA.getValue(ModelType.THERMAL);
            double tempB = nodeB.getValue(ModelType.THERMAL);
            double delta = tempA - tempB;

            graphics.drawString(this.font,
                    String.format("Temp A: %.2f°C", tempA),
                    panelX + 10, panelY + 50, 0xFFFFFFFF);
            graphics.drawString(this.font,
                    String.format("Temp B: %.2f°C", tempB),
                    panelX + 10, panelY + 65, 0xFFFFFFFF);
            graphics.drawString(this.font,
                    String.format("ΔT: %.2f°C", delta),
                    panelX + 10, panelY + 80, 0xFFFFD700);
            graphics.drawString(this.font,
                    String.format("Heat Flow: %.2f W", delta * thermalLink.getConductance()),
                    panelX + 10, panelY + 95, 0xFFFF4500);
        }
    }

    private void renderConnectionLine(GuiGraphics graphics, NodeWidget from, NodeWidget to) {
        int x1 = from.getX() + from.getWidth() / 2;
        int y1 = from.getY() + from.getHeight() / 2;
        int x2 = to.getX() + to.getWidth() / 2;
        int y2 = to.getY() + to.getHeight() / 2;

        // Draw line representing thermal link
        drawThickLine(graphics, x1, y1, x2, y2, 0xFFFF4500, 3);

        // Draw conductance label
        int midX = (x1 + x2) / 2;
        int midY = (y1 + y2) / 2 - 15;
        graphics.drawCenteredString(this.font,
                String.format("G = %.1f W/K", thermalLink.getConductance()),
                midX, midY, 0xFFFFFFFF);
    }

    private void renderHeatFlow(GuiGraphics graphics) {
        // Render animated heat flow arrows if solved
        int sourceX = widgetA.getX() - 50;
        int sourceY = widgetA.getY() + widgetA.getHeight() / 2;
        int targetX = widgetA.getX();
        int targetY = widgetA.getY() + widgetA.getHeight() / 2;

        drawArrow(graphics, sourceX, sourceY, targetX, targetY, 0xFFFFD700);

        // Draw source power label
        graphics.drawString(this.font,
                String.format("%.0f W", heatSource.getFlux()),
                sourceX - 30, sourceY - 10, 0xFFFFD700);
    }

    private void renderResults(GuiGraphics graphics) {
        // Additional results visualization at the bottom
        int y = this.height - 140;
        graphics.drawCenteredString(this.font, "Results:", this.width / 2, y, 0xFFFFD700);

        double tempA = nodeA.getValue(ModelType.THERMAL);
        double delta = tempA - nodeB.getValue(ModelType.THERMAL);

        String result = String.format("Temperature difference: %.2f°C (Expected: 10°C)", delta);
        int color = Math.abs(delta - 10.0) < 0.01 ? 0xFF00FF00 : 0xFFFFFFFF;
        graphics.drawCenteredString(this.font, result, this.width / 2, y + 15, color);
    }

    private void drawThickLine(GuiGraphics graphics, int x1, int y1, int x2, int y2, int color, int thickness) {
        for (int i = -thickness/2; i <= thickness/2; i++) {
            graphics.fill(x1, y1 + i, x2, y2 + i, color);
        }
    }

    private void drawArrow(GuiGraphics graphics, int x1, int y1, int x2, int y2, int color) {
        // Draw line
        drawThickLine(graphics, x1, y1, x2, y2, color, 2);

        // Draw arrowhead
        int arrowSize = 8;
        graphics.fill(x2 - arrowSize, y2 - arrowSize/2, x2, y2, color);
        graphics.fill(x2 - arrowSize, y2, x2, y2 + arrowSize/2, color);
    }

    @Override
    public boolean isPauseScreen() {
        return false;
    }
}
