package com.rae.formicapi.fondation.math.data;

import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * @param interpolationType "linear", "cubic", "smooth"
 */
public record SplineBased2DFunction(List<IsoLine> isoLines, boolean clamp, InterpolationType interpolationType) {

    public SplineBased2DFunction(List<IsoLine> isoLines, boolean clamp, InterpolationType interpolationType) {
        this.isoLines = new ArrayList<>(isoLines);
        // Sort iso-lines by value for efficient lookup
        this.isoLines.sort(Comparator.comparing(IsoLine::getValue));
        this.clamp = clamp;
        this.interpolationType = interpolationType;
    }

    //TODO replace Vec2 by a float[]
    //TODO replace the string type by enums
    //TODO add the necessary work for synchronisation work (split and merge)

    //TODO fix evaluate : if it's outside the table it will fail because the point we need are not the closest but
    // the one that have a continued line that is close to us
    // OR we could use it to generate a 2D grid that has the correct derivative for interpolations
    // (2d grid of both value and 2d gradient of 1st and 2nd order)
    /**
     * Evaluates the function at point (x, y)
     * Strategy:
     * 1. Find the closest point on each iso-line to (x, y)
     * 2. Use the distances and iso-line values to interpolate
     */
    public float evaluate(float x, float y) {
        if (isoLines.isEmpty()) {
            throw new IllegalStateException("No iso-lines defined");
        }

        // Find closest points on each iso-line
        List<IsoLineProximity> proximities = new ArrayList<>();

        //This is a quadratic cost !!!!
        for (IsoLine isoLine : isoLines) {
            PointOnSpline closest = isoLine.findClosestPoint(x, y);
            proximities.add(new IsoLineProximity(isoLine, closest));
        }

        // Sort by distance to find nearest iso-lines
        proximities.sort(Comparator.comparing(p -> p.closestPoint.distanceSquared));

        // Use the two nearest iso-lines for interpolation
        if (proximities.size() == 1) {
            return proximities.get(0).isoLine.getValue();
        }

        IsoLineProximity nearest     = proximities.get(0);
        IsoLineProximity nextNearest = proximities.get(1);

        // Interpolate between the two nearest iso-lines
        return interpolateBetweenIsoLines(x, y, nearest, nextNearest);
    }

    private float interpolateBetweenIsoLines(float x, float y,
                                             IsoLineProximity p1,
                                             IsoLineProximity p2) {
        float d1 = (float) Math.sqrt(p1.closestPoint.distanceSquared);
        float d2 = (float) Math.sqrt(p2.closestPoint.distanceSquared);

        if (d1 + d2 < 1e-6f) {
            // Point lies exactly on an iso-line
            return p1.isoLine.getValue();
        }

        // Inverse distance weighting
        float w1 = d2 / (d1 + d2);
        float w2 = d1 / (d1 + d2);

        float v1 = p1.isoLine.getValue();
        float v2 = p2.isoLine.getValue();

        return switch (interpolationType) {
            case CUBIC -> cubicInterpolate(v1, v2, w2);
            case SMOOTH -> smoothstepInterpolate(v1, v2, w2);
            default -> // "linear"
                    v1 * w1 + v2 * w2;
        };
    }

    private float cubicInterpolate(float v1, float v2, float t) {
        float t2 = t * t;
        float t3 = t2 * t;
        return v1 * (1 - 3 * t2 + 2 * t3) + v2 * (3 * t2 - 2 * t3);
    }

    private float smoothstepInterpolate(float v1, float v2, float t) {
        t = t * t * (3.0f - 2.0f * t);
        return v1 * (1 - t) + v2 * t;
    }

    /**
     * Represents an iso-line (contour line) in 2D space
     */
    public static class IsoLine {
        private final float        value;
        private final List<Vec2>   controlPoints;
        private final String       splineType;
        private       CachedSpline cachedSpline;

        public IsoLine(float value, List<Vec2> controlPoints, String splineType) {
            this.value = value;
            this.controlPoints = new ArrayList<>(controlPoints);
            this.splineType = splineType;
            this.cachedSpline = null;
        }

        public float getValue() {
            return value;
        }

        public List<Vec2> getControlPoints() {
            return controlPoints;
        }

        public String getSplineType() {
            return splineType;
        }

        private CachedSpline getSpline() {
            if (cachedSpline == null) {
                cachedSpline = new CachedSpline(controlPoints, splineType);
            }
            return cachedSpline;
        }

        /**
         * Finds the closest point on this iso-line to (x, y)
         */
        private PointOnSpline findClosestPoint(float x, float y) {
            CachedSpline spline    = getSpline();
            float        bestT     = getBestT(x, y, spline);

            // Refine using Newton's method or gradient descent
            // (simplified version here)
            float refinedT     = refineClosestPoint(x, y, bestT, spline);
            Vec2  refinedPoint = spline.evaluate(refinedT);
            float refinedDistSq = (refinedPoint.x - x) * (refinedPoint.x - x) +
                    (refinedPoint.y - y) * (refinedPoint.y - y);

            return new PointOnSpline(refinedPoint, refinedDistSq, refinedT);
        }

        private float getBestT(float x, float y, CachedSpline spline) {
            float        minDistSq = Float.MAX_VALUE;
            float        bestT     = 0;

            // Sample the spline at regular intervals
            int samples = Math.max(50, controlPoints.size() * 10);
            for (int i = 0; i <= samples; i++) {
                float t     = (float) i / samples;
                Vec2  point = spline.evaluate(t);
                float distSq = (point.x - x) * (point.x - x) +
                        (point.y - y) * (point.y - y);

                if (distSq < minDistSq) {
                    minDistSq = distSq;
                    bestT = t;
                }
            }
            return bestT;
        }

        private float refineClosestPoint(float x, float y, float initialT, CachedSpline spline) {
            float t             = initialT;
            float epsilon       = 0.001f;
            int   maxIterations = 10;

            for (int iter = 0; iter < maxIterations; iter++) {
                Vec2 p  = spline.evaluate(t);
                Vec2 dp = spline.evaluateDerivative(t);

                // Gradient of distance squared
                float grad = 2 * ((p.x - x) * dp.x + (p.y - y) * dp.y);

                if (Math.abs(grad) < epsilon) break;

                // Simple gradient descent step
                float stepSize = 0.01f;
                float newT     = t - stepSize * grad;
                newT = Math.max(0, Math.min(1, newT));

                if (Math.abs(newT - t) < epsilon) break;
                t = newT;
            }

            return Math.max(0, Math.min(1, t));
        }
    }

    /**
     * Cached spline evaluation
     */
    private record CachedSpline(List<Vec2> controlPoints, String type) {

        public Vec2 evaluate(float t) {
            if (controlPoints.size() < 2) {
                return controlPoints.get(0);
            }

            return switch (type) {
                case "cubic" -> evaluateCubicBezier(t);
                default -> evaluateCatmullRom(t);
            };
        }

        public Vec2 evaluateDerivative(float t) {
            float h  = 0.001f;
            Vec2  p1 = evaluate(Math.max(0, t - h));
            Vec2  p2 = evaluate(Math.min(1, t + h));
            return new Vec2((p2.x - p1.x) / (2 * h), (p2.y - p1.y) / (2 * h));
        }

        private Vec2 evaluateCatmullRom(float t) {
            int n = controlPoints.size();
            if (n == 0) throw new IllegalStateException("No control points");
            if (n == 1) return controlPoints.get(0);

            float scaledT = t * (n - 1);
            int   segment = Math.min((int) scaledT, n - 2);
            float localT  = scaledT - segment;

            Vec2 p0 = segment > 0 ? controlPoints.get(segment - 1) : controlPoints.get(0);
            Vec2 p1 = controlPoints.get(segment);
            Vec2 p2 = controlPoints.get(segment + 1);
            Vec2 p3 = segment + 2 < n ? controlPoints.get(segment + 2) : controlPoints.get(n - 1);

            return catmullRomPoint(p0, p1, p2, p3, localT);
        }

        private Vec2 catmullRomPoint(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, float t) {
            float t2 = t * t;
            float t3 = t2 * t;

            float x = 0.5f * ((2 * p1.x) +
                    (-p0.x + p2.x) * t +
                    (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
                    (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3);

            float y = 0.5f * ((2 * p1.y) +
                    (-p0.y + p2.y) * t +
                    (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
                    (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3);

            return new Vec2(x, y);
        }

        private Vec2 evaluateCubicBezier(float t) {
            // Simplified: treats control points as cubic Bezier segments
            int n = controlPoints.size();
            if (n < 4) return evaluateCatmullRom(t);

            int   segments = (n - 1) / 3;
            float scaledT  = t * segments;
            int   segment  = Math.min((int) scaledT, segments - 1);
            float localT   = scaledT - segment;

            int  baseIdx = segment * 3;
            Vec2 p0      = controlPoints.get(baseIdx);
            Vec2 p1      = controlPoints.get(baseIdx + 1);
            Vec2 p2      = controlPoints.get(baseIdx + 2);
            Vec2 p3      = controlPoints.get(Math.min(baseIdx + 3, n - 1));

            return cubicBezierPoint(p0, p1, p2, p3, localT);
        }

        private Vec2 cubicBezierPoint(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, float t) {
            float u   = 1 - t;
            float tt  = t * t;
            float uu  = u * u;
            float uuu = uu * u;
            float ttt = tt * t;

            float x = uuu * p0.x + 3 * uu * t * p1.x + 3 * u * tt * p2.x + ttt * p3.x;
            float y = uuu * p0.y + 3 * uu * t * p1.y + 3 * u * tt * p2.y + ttt * p3.y;

            return new Vec2(x, y);
        }
    }

    public enum InterpolationType {
        LINEAR,
        CUBIC,
        SMOOTH
    }

    /**
     * Simple 2D vector class
     */
    public record Vec2(float x, float y) {
        public static final Codec<Vec2> CODEC = RecordCodecBuilder.create(instance ->
                instance.group(
                        Codec.FLOAT.fieldOf("x").forGetter(v -> v.x),
                        Codec.FLOAT.fieldOf("y").forGetter(v -> v.y)
                ).apply(instance, Vec2::new)
        );

    }

    private record PointOnSpline(Vec2 point, float distanceSquared, float t) {
    }

    private record IsoLineProximity(IsoLine isoLine, PointOnSpline closestPoint) {
    }

    /**
     * Builder for creating spline-based functions
     */
    public static class Builder {
        private final List<IsoLine>     isoLines          = new ArrayList<>();
        private       boolean           clamp             = true;
        private       InterpolationType interpolationType = InterpolationType.LINEAR;

        public Builder addIsoLine(float value, List<Vec2> controlPoints) {
            return addIsoLine(value, controlPoints, "catmull_rom");
        }

        public Builder addIsoLine(float value, List<Vec2> controlPoints, String splineType) {
            isoLines.add(new IsoLine(value, controlPoints, splineType));
            return this;
        }

        public Builder setClamp(boolean clamp) {
            this.clamp = clamp;
            return this;
        }

        public Builder setInterpolationType(InterpolationType type) {
            this.interpolationType = type;
            return this;
        }

        public SplineBased2DFunction build() {
            return new SplineBased2DFunction(isoLines, clamp, interpolationType);
        }
    }
}