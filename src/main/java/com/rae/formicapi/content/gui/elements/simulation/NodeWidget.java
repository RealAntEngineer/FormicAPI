package com.rae.formicapi.content.gui.elements.simulation;

import com.rae.formicapi.fondation.simulation.nodal.ModelType;
import com.rae.formicapi.fondation.simulation.nodal.core.Node;
import net.createmod.catnip.gui.widget.AbstractSimiWidget;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiGraphics;
import org.lwjgl.system.NonnullDefault;

@NonnullDefault
public class NodeWidget extends AbstractSimiWidget {
    private static final int NODE_SIZE      = 40;
    private static final int VALUE_OFFSET_Y = 15;
    private static final int DOMAIN_SPACING = 12;
    public Node node;

    public NodeWidget(Node node, int x, int y) {
        super(x, y);
        this.node = node;
        this.width = NODE_SIZE;
        this.height = NODE_SIZE;
    }

    @Override
    protected void renderWidget(GuiGraphics graphics, int mouseX, int mouseY, float partialTicks) {
        super.renderWidget(graphics, mouseX, mouseY, partialTicks);

        // Determine if node is hovered
        boolean isHovered = mouseX >= getX() && mouseX <= getX() + width &&
                mouseY >= getY() && mouseY <= getY() + height;

        // Draw node circle/shape
        int centerX = getX() + width / 2;
        int centerY = getY() + height / 2;
        int radius  = NODE_SIZE / 2;

        // Background circle
        int nodeColor = isHovered ? 0xFF4A90E2 : 0xFF2C5F8D;
        graphics.fill(centerX - radius, centerY - radius,
                centerX + radius, centerY + radius, nodeColor);

        // Border
        int borderColor = isActive() ? 0xFFFFD700 : 0xFF1A3A52;
        drawCircleBorder(graphics, centerX, centerY, radius, borderColor, 2);

        // Render node label (if exists)
        /*if (node.getLabel() != null && !node.getLabel().isEmpty()) {
            graphics.drawCenteredString(Minecraft.getInstance().font,
                    node.getLabel(),
                    centerX,
                    getY() - 12,
                    0xFFFFFFFF);
        }*/

        // Render the value string per domain
        int yOffset = getY() + height + VALUE_OFFSET_Y;

        for (ModelType model : node.getDomains()) {
            double value = node.getValue(model);

            String displayText = /*model.getDisplayName() + ": " +*/ formatValue(model, value);//use components ?

            // Choose color based on domain type
            int textColor = getDomainColor(model);

            // Draw the text
            graphics.drawString(Minecraft.getInstance().font,
                    displayText,
                    getX(),
                    yOffset,
                    textColor);

            yOffset += DOMAIN_SPACING;
        }

        // Optional: Draw connections/edges to other nodes
        if (isHovered) {
            renderConnections(graphics, mouseX, mouseY);
        }
    }

    /**
     * Check if this node is currently active/selected
     */
    public boolean isActive() {
        // Implement your active state logic
        return false;
    }

    /**
     * Draw a circle border
     */
    private void drawCircleBorder(GuiGraphics graphics, int centerX, int centerY,
                                  int radius, int color, int thickness) {
        for (int t = 0; t < thickness; t++) {
            int r = radius + t;
            // Simple circle approximation using rectangles
            for (int angle = 0; angle < 360; angle += 5) {
                double rad = Math.toRadians(angle);
                int    x   = centerX + (int) (r * Math.cos(rad));
                int    y   = centerY + (int) (r * Math.sin(rad));
                graphics.fill(x, y, x + 1, y + 1, color);
            }
        }
    }

    /**
     * Format the value based on the model type
     */
    private String formatValue(ModelType model, Object value) {
        if (value instanceof Double || value instanceof Float) {
            return String.format("%.2f", ((Number) value).doubleValue());
        } /*else if (value instanceof Vector3) {
            Vector3 vec = (Vector3) value;
            return String.format("(%.1f, %.1f, %.1f)", vec.x, vec.y, vec.z);
        }*/ else if (value instanceof Number) {
            return String.valueOf(value);
        }
        return value.toString();
    }

    /**
     * Get color for each domain type
     */
    private int getDomainColor(ModelType model) {
        return switch (model) {
            case MECHANICAL -> 0xFF00FF00;  // Green
            case THERMAL -> 0xFFFF4500;     // Orange-Red
            //case ELECTRICAL -> 0xFFFFFF00;  // Yellow
            case HYDRAULIC -> 0xFF00BFFF;       // Deep Sky Blue
            //case MAGNETIC -> 0xFFFF00FF;    // Magenta
            default -> 0xFFFFFFFF;          // White
        };
    }

    /**
     * Render connections to other nodes (if applicable)
     */
    private void renderConnections(GuiGraphics graphics, int mouseX, int mouseY) {
        // Implement edge rendering if your nodes have connections
        // Example: draw lines to connected nodes
    }
}