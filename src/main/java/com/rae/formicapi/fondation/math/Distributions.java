package com.rae.formicapi.fondation.math;

import java.util.Random;

public class Distributions {

    // ─── Poisson ────────────────────────────────────────────────────────────────

    private static final int POISSON_STEP = 500;

    public static int nextPoisson(Random rd, double lambda) {
        if (lambda < 30)  return poissonInverse(rd, lambda);
        if (lambda < 700) return poissonKnuth(rd, lambda);
        return poissonJunhao(rd, lambda);
    }

    private static int poissonInverse(Random rd, double lambda) {
        double p = Math.exp(-lambda), s = p, u = rd.nextDouble();
        int x = 0;
        while (u > s) { p *= lambda / ++x; s += p; }
        return x;
    }

    private static int poissonKnuth(Random rd, double lambda) {
        double limit = Math.exp(-lambda), product = 1.0;
        int k = 0;
        do { k++; product *= rd.nextDouble(); } while (product > limit);
        return k - 1;
    }

    private static int poissonJunhao(Random rd, double lambda) {
        double lambdaLeft = lambda, p = 1.0;
        int k = 0;
        do {
            k++;
            p *= rd.nextDouble();
            while (p < 1 && lambdaLeft > 0) {
                if (lambdaLeft > POISSON_STEP) { p *= Math.exp(POISSON_STEP); lambdaLeft -= POISSON_STEP; }
                else { p *= Math.exp(lambdaLeft); lambdaLeft = 0; }
            }
        } while (p > 1);
        return k - 1;
    }

    // ─── Binomial ───────────────────────────────────────────────────────────────
    // B(n, p): number of successes in n independent Bernoulli trials

    public static int nextBinomial(Random rd, int n, double p) {
        if (n <= 0 || p <= 0) return 0;
        if (p >= 1)           return n;
        double np = n * p;
        if (n < 30)           return binomialDirect(rd, n, p);     // small n: simulate trials
        if (np < 30)          return binomialInverse(rd, n, p);    // sparse: inverse transform
        return binomialNormal(rd, n, p);                           // large np: normal approx
    }

    // Simulate each trial — exact but O(n)
    private static int binomialDirect(Random rd, int n, double p) {
        int successes = 0;
        for (int i = 0; i < n; i++) if (rd.nextDouble() < p) successes++;
        return successes;
    }

    // Inverse transform — best when np is small
    private static int binomialInverse(Random rd, int n, double p) {
        double q = 1 - p;
        double prob = Math.pow(q, n); // P(X=0)
        double s = prob;
        double u = rd.nextDouble();
        int x = 0;
        while (u > s && x < n) {
            prob *= ((n - x) * p) / ((x + 1) * q); // recurrence P(X=k+1)
            s += prob;
            x++;
        }
        return x;
    }

    // Normal approximation — best when np and n(1-p) are both large
    private static int binomialNormal(Random rd, int n, double p) {
        double mean = n * p;
        double std  = Math.sqrt(n * p * (1 - p));
        return (int) Math.round(mean + std * nextGaussian(rd));
    }

    // ─── Normal (Gaussian) ──────────────────────────────────────────────────────
    // Box-Muller: exact, generates 2 independent samples per call

    private static double spareGaussian;
    private static boolean hasSpare = false;

    public static double nextGaussian(Random rd) {
        if (hasSpare) { hasSpare = false; return spareGaussian; }
        double u, v, s;
        do {
            u = rd.nextDouble() * 2 - 1;
            v = rd.nextDouble() * 2 - 1;
            s = u * u + v * v;
        } while (s >= 1 || s == 0);
        double factor = Math.sqrt(-2.0 * Math.log(s) / s);
        spareGaussian = v * factor;
        hasSpare = true;
        return u * factor;
    }

    public static double nextGaussian(Random rd, double mean, double std) {
        return mean + std * nextGaussian(rd);
    }

    // ─── Exponential ────────────────────────────────────────────────────────────
    // Closed-form inverse CDF: X = -ln(U) / λ

    public static double nextExponential(Random rd, double lambda) {
        return -Math.log(1 - rd.nextDouble()) / lambda;
    }

    // ─── Geometric ──────────────────────────────────────────────────────────────
    // Closed-form: number of trials until first success
    // X = floor(ln(U) / ln(1-p))

    public static int nextGeometric(Random rd, double p) {
        return (int) Math.floor(Math.log(rd.nextDouble()) / Math.log(1 - p));
    }

    // ─── Gamma ──────────────────────────────────────────────────────────────────
    // Marsaglia-Tsang algorithm — works for α ≥ 1, handles α < 1 via reduction

    public static double nextGamma(Random rd, double alpha, double beta) {
        if (alpha < 1) {
            // Reduction: Gamma(α) = Gamma(α+1) * U^(1/α)
            return nextGamma(rd, alpha + 1, beta) * Math.pow(rd.nextDouble(), 1.0 / alpha);
        }
        double d = alpha - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9 * d);
        while (true) {
            double x, v;
            do { x = nextGaussian(rd); v = 1 + c * x; } while (v <= 0);
            v = v * v * v;
            double u = rd.nextDouble();
            if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v / beta;
            if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v / beta;
        }
    }

    // ─── Beta ───────────────────────────────────────────────────────────────────
    // Via ratio of two Gamma samples: X ~ Gamma(α), Y ~ Gamma(β) → X/(X+Y) ~ Beta(α,β)

    public static double nextBeta(Random rd, double alpha, double beta) {
        double x = nextGamma(rd, alpha, 1);
        double y = nextGamma(rd, beta,  1);
        return x / (x + y);
    }
}
