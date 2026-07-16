package com.rae.formicapi.foundation.math.pde.ast;

import com.rae.formicapi.foundation.math.pde.FieldType;
import com.rae.formicapi.foundation.math.pde.SymbolBinding;
import com.rae.formicapi.foundation.math.pde.SymbolRole;
import org.checkerframework.common.value.qual.IntRange;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Base type for every node in a parsed equation's abstract syntax tree.
 *
 * <p>An {@code Expression} tree is built once by {@link #parseExpression} and is immutable
 * from then on: every transformation (see {@link #expand()}, or {@code ScalarAlgebra}'s
 * passes) returns a new tree rather than mutating this one. Concrete node types are
 * {@link ConstantExpression}, {@link VariableExpression}, {@link UnaryExpression},
 * {@link BinaryExpression}, {@link VectorExpression}, {@link MatrixExpression}, and
 * {@link ComponentExpression}.
 *
 * <p>Every node carries two pieces of structural bookkeeping, both fixed at construction and
 * unaffected by later transformations of the tree it's part of:
 * <ul>
 *     <li>{@link #dimensions()} — the spatial dimensionality (1, 2, or 3) this expression was
 *     parsed/built for. All operands of a binary operation must share the same value; see
 *     {@link BinaryExpression}.</li>
 *     <li>{@link #getDepth()} — how deeply nested this node is below the tree's leaves
 *     (leaves are depth 0). Used by {@link #appearAfter} to break ties when canonically
 *     ordering the operands of commutative operators, and checked against {@link #MAX_DEPTH}
 *     during parsing to fail fast on pathological input instead of recursing forever.</li>
 * </ul>
 *
 * <p>Implementations must also satisfy:
 * <ul>
 *     <li>{@link #resultType()} is consistent with how the node was constructed — e.g. a
 *     {@link BinaryExpression} validates this eagerly against its operator at construction
 *     time, so a type mismatch throws {@link FieldType.TypeMismatchException} immediately
 *     rather than when {@code resultType()} is later called.</li>
 *     <li>{@code equals}/{@code hashCode} are structural (two independently-built trees with
 *     the same shape and content must be equal), since {@code ScalarAlgebra}'s term-grouping
 *     relies on this to merge like terms.</li>
 * </ul>
 */
public abstract class Expression {
    /**
     * Maximum recursion depth {@link #parseExpression} will descend to before giving up.
     * Guards against unbounded recursion on malformed input (e.g. unbalanced nested
     * parentheses that keep re-triggering a parenthesis group at every level).
     */
    public static final int MAX_DEPTH = 100;
    private final       int dimensions;
    private final       int depth;


    Expression(@IntRange(from = 1) int dimensions) {
        this(dimensions, 0);
    }

    Expression(@IntRange(from = 1) int dimensions, @IntRange(from = 0) int depth) {
        this.dimensions = dimensions;
        this.depth = depth;
    }

    /**
     * Parses a textual expression into an {@link Expression} tree.
     *
     * <p>Recognizes numeric literals, symbol references looked up in {@code symbols}, unary
     * operators applied to a parenthesized operand (e.g. {@code grad(T)}), parenthesized
     * groups, implicit multiplication between a value and a directly-following parenthesis
     * (e.g. {@code 2(a+b)} or {@code (a)(b)}), and binary operators combined according to
     * {@link BinaryOperators} priority and associativity.
     *
     * <p>This method recurses into itself once per parenthesized group or unary-operator
     * argument, with {@code depth} incremented each time; top-level callers should pass
     * {@code depth = 0}.
     *
     * @param text       expression text to parse; leading/trailing whitespace is trimmed,
     *                   internal whitespace separates tokens
     * @param symbols    known symbols, keyed by name, that identifiers in {@code text} may
     *                   reference
     * @param depth      current recursion depth; pass {@code 0} at the top level
     * @param dimensions spatial dimensionality to assign to every node built from this parse
     * @return the parsed expression tree
     * @throws RuntimeException if {@code depth} exceeds {@link #MAX_DEPTH}, the text is empty,
     *                          contains an identifier that isn't a known symbol or operator,
     *                          contains an unparseable numeric literal, contains an
     *                          unrecognized binary-operator character, or has unmatched
     *                          parentheses (see {@link #groupByParenthesis})
     */
    public static Expression parseExpression(String text, Map<String, SymbolBinding> symbols, @IntRange(from = 0) int depth, @IntRange(from = 1) int dimensions) {

        if (depth > MAX_DEPTH)
            throw new RuntimeException("Max depth reached");

        text = text.trim();

        List<Expression>      operands  = new ArrayList<>();
        List<BinaryOperators> operators = new ArrayList<>();

        String remaining = text;

        while (true) {
            remaining = remaining.trim();

            Expression operand;

            if (remaining.startsWith("(")) {
                String[] splited = groupByParenthesis(remaining);
                operand = parseExpression(splited[0], symbols, depth + 1, dimensions);
                remaining = splited[1];

            } else {
                String identifier = remaining;
                String rest       = "";

                for (int i = 0; i < remaining.length(); i++) {
                    char c = remaining.charAt(i);
                    if (c == ' ') {
                        identifier = remaining.substring(0, i);
                        rest = remaining.substring(i + 1);
                        break;
                    } else if (c == '(' || BinaryOperators.isOperatorChar(c)) {
                        identifier = remaining.substring(0, i);
                        rest = remaining.substring(i);
                        break;
                    }
                }

                if (identifier.isEmpty())
                    throw new RuntimeException("Expression is empty");

                if (Character.isDigit(identifier.charAt(0))) {
                    try {
                        operand = new ConstantExpression(Double.parseDouble(identifier), dimensions);
                    } catch (NumberFormatException e) {
                        throw new RuntimeException("Failed to parse value " + identifier + " in expression " + text, e);
                    }

                } else if (symbols.containsKey(identifier)) {
                    operand = new VariableExpression(symbols.get(identifier), dimensions);

                } else {
                    Optional<UnaryOperators> optionalUnary = UnaryOperators.parse(identifier);
                    if (optionalUnary.isPresent()) {
                        String[] splited = groupByParenthesis(rest);
                        operand = new UnaryExpression(optionalUnary.get(), parseExpression(splited[0], symbols, depth + 1, dimensions));
                        rest = splited[1];
                    } else {
                        throw new RuntimeException("Unable to parse identifier : " + identifier);
                    }
                }
                remaining = rest;
            }

            // implicit multiplication: operand directly followed by '(' , e.g. "2(3+4)"
            remaining = remaining.trim();
            while (remaining.startsWith("(")) {
                String[]   splited = groupByParenthesis(remaining);
                Expression right   = parseExpression(splited[0], symbols, depth + 1, dimensions);
                operand = new BinaryExpression(BinaryOperators.MULTIPLY, operand, right);
                remaining = splited[1].trim();
            }

            operands.add(operand);

            if (remaining.isEmpty())
                break;

            Optional<BinaryOperators> optionalBinary = BinaryOperators.parse(remaining.charAt(0));
            if (optionalBinary.isEmpty())
                throw new RuntimeException("Unable to parse binary operator : " + remaining.charAt(0));

            operators.add(optionalBinary.get());
            remaining = remaining.substring(1);
        }

        return buildTreeByPrecedence(operands, operators);
    }

    /**
     * Assembles a flat list of operands and the binary operators between them into a tree,
     * respecting each operator's {@link BinaryOperators#priority} and associativity
     * ({@link BinaryOperators#rightMerging}).
     *
     * <p>Processes one priority level at a time, from highest to lowest, so that e.g.
     * {@code POWER} binds before {@code MULTIPLY} before {@code ADD}/{@code SUBTRACT}. Within
     * a priority level, right-associative operators (currently just {@code POWER}) are
     * combined right-to-left; all others are combined left-to-right.
     *
     * @param operands  operands in left-to-right order; must have exactly one more element
     *                  than {@code operators}
     * @param operators binary operators in left-to-right order, one between each adjacent
     *                  pair of operands
     * @return the single expression tree combining all operands
     */
    private static Expression buildTreeByPrecedence(List<Expression> operands, List<BinaryOperators> operators) {
        List<Expression>      outOperands  = new ArrayList<>(operands);
        List<BinaryOperators> outOperators = new ArrayList<>(operators);

        int maxPriority = 0;
        for (BinaryOperators op : outOperators) {
            maxPriority = Math.max(maxPriority, op.priority);
        }

        for (int priority = maxPriority; priority >= 0; priority--) {

            boolean rightAssociative = isRightAssociativeLevel(outOperators, priority);

            if (rightAssociative) {
                // scan right-to-left, combine as we go
                int i = outOperators.size() - 1;
                while (i >= 0) {
                    if (outOperators.get(i).priority == priority) {
                        Expression combined = new BinaryExpression(
                                outOperators.get(i),
                                outOperands.get(i),
                                outOperands.get(i + 1));

                        outOperands.set(i, combined);
                        outOperands.remove(i + 1);
                        outOperators.remove(i);
                    }
                    i--;
                }
            } else {
                // scan left-to-right, combine as we go
                int i = 0;
                while (i < outOperators.size()) {
                    if (outOperators.get(i).priority == priority) {
                        Expression combined = new BinaryExpression(
                                outOperators.get(i),
                                outOperands.get(i),
                                outOperands.get(i + 1));

                        outOperands.set(i, combined);
                        outOperands.remove(i + 1);
                        outOperators.remove(i);
                        // don't advance: next operator shifted into position i
                    } else {
                        i++;
                    }
                }
            }
        }

        return outOperands.getFirst();
    }

    /**
     * Splits a string starting with {@code (} into the contents of its leading parenthesized
     * group and whatever text follows the matching close paren.
     *
     * <pre>
     * groupByParenthesis("(a + b * c ((()))) + 1") -> {"a + b * c ((()))", " + 1"}
     * </pre>
     *
     * @param text text beginning with {@code (}
     * @return a 2-element array: {@code [0]} is the group's inner content (parentheses
     * stripped), {@code [1]} is everything after the matching close paren (empty string if
     * nothing follows)
     * @throws RuntimeException if {@code text} doesn't start with {@code (}, is a single
     *                          character, or has no matching close paren
     */
    public static String[] groupByParenthesis(String text) {

        if (!text.startsWith("(") || text.length() == 1) {
            throw new RuntimeException("Unmatched parenthesis at start");
        }

        int parenthesisCount = 1;
        for (int i = 1; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == '(') {
                parenthesisCount++;
            } else if (c == ')') {
                parenthesisCount--;
            }

            if (parenthesisCount <= 0) {
                if (i <= text.length() - 1) {
                    return new String[]{text.substring(1, i), text.substring(i + 1)};
                } else {
                    return new String[]{text.substring(1, text.length() - 1), ""};
                }
            }

        }

        throw new RuntimeException("Unmatched parenthesis");
    }

    /**
     * Whether the priority level being processed by {@link #buildTreeByPrecedence} should be
     * combined right-to-left.
     *
     * @param operators the full operator list being processed
     * @param priority  the priority level currently being resolved
     * @return {@code true} if an operator at this priority level is right-associative; if no
     * operator in {@code operators} is at this priority level, returns {@code false} (the
     * value is unused in that case)
     */
    private static boolean isRightAssociativeLevel(List<BinaryOperators> operators, int priority) {
        for (BinaryOperators op : operators) {
            if (op.priority == priority) {
                return op.rightMerging;
            }
        }
        return false; // doesn't matter, no operator at this level
    }

    /**
     * The spatial dimensionality (1, 2, or 3) this expression and its subtree were built for.
     * Fixed at construction; all operands combined by a {@link BinaryExpression} must agree
     * on this value.
     */
    public final int dimensions() {
        return dimensions;
    }

    /**
     * Wraps this expression in a {@link ComponentExpression} that symbolically extracts a
     * single vector component, without requiring this expression to already be a
     * {@link VectorExpression}. If this expression already stores its components directly
     * (a {@link VectorExpression}), the stored sub-expression is returned instead of a new
     * wrapper node — see {@link ComponentExpression#ofVector}.
     *
     * @param axis zero-based component index
     * @return an expression representing this expression's {@code axis}-th component
     * @throws FieldType.TypeMismatchException if this expression's {@link #resultType()} is
     *                                          not {@code VECTOR} or {@code MATRIX}
     */
    public Expression componentAt(int axis) {
        return ComponentExpression.ofVector(this, axis);
    }

    /**
     * Wraps this expression in a {@link ComponentExpression} that symbolically extracts a
     * single matrix entry, without requiring this expression to already be a
     * {@link MatrixExpression}. If this expression already stores its entries directly (a
     * {@link MatrixExpression}), the stored sub-expression is returned instead of a new
     * wrapper node — see {@link ComponentExpression#ofMatrix}.
     *
     * @param row zero-based row index
     * @param col zero-based column index
     * @return an expression representing this expression's {@code (row, col)} entry
     * @throws FieldType.TypeMismatchException if this expression's {@link #resultType()} is
     *                                          not {@code MATRIX}
     */
    public Expression componentAt(int row, int col) {
        return ComponentExpression.ofMatrix(this, row, col);
    }

    /**
     * The field type ({@code SCALAR}, {@code VECTOR}, or {@code MATRIX}) this expression
     * evaluates to.
     *
     * @throws FieldType.TypeMismatchException if this node's operator/shape is incompatible
     *                                          with its operands' types (nodes validate this
     *                                          eagerly at construction, so in practice this is
     *                                          thrown when the node is built, not here)
     */
    public abstract FieldType resultType();

    /**
     * Renders this expression as a human-readable string using infix notation for binary
     * operators (parenthesizing only where operator priority/associativity requires it) and
     * {@code operator(operand)} notation for unary operators.
     */
    public abstract String prettyPrint();

    /**
     * Rewrites this expression into an equivalent but more explicit form: distributing
     * multiplication/division over addition/subtraction, applying linearity and the product
     * rule for differential operators, lowering vector/tensor-valued operators (grad, div,
     * dot/cross/outer product) into per-component scalar expressions, and collapsing composed
     * differential operators into their simplified form (e.g. {@code div(grad(f))} into a
     * Laplacian, {@code ddx(ddx(f))} into a second derivative).
     *
     * <p>Idempotent: calling {@code expand()} on an already-expanded expression returns an
     * equivalent expression. Does <strong>not</strong> combine like terms or extract common
     * factors — those are separate, arithmetic-level passes (see {@code ScalarAlgebra}) that
     * this method does not perform.
     *
     * @return an equivalent expression, rewritten as described above
     */
    public abstract Expression expand();

    /**
     * Whether this expression's value can change with respect to time — i.e. whether it's
     * meaningful to apply a time derivative ({@code ddt}, {@code d2dt2}) to it. Determined by
     * the {@link SymbolRole} of the symbols this
     * expression is built from; see
     * {@link SymbolRole#isTimeDifferentiable()}.
     */
    public abstract boolean isTimeDifferentiable();

    /**
     * Whether this expression's value can vary spatially — i.e. whether it's meaningful to
     * apply a spatial differential operator ({@code grad}, {@code div}, {@code lap}, axis
     * derivatives) to it. Determined by the
     * {@link SymbolRole} of the symbols this expression
     * is built from; see
     * {@link SymbolRole#isSpaceDifferentiable()}.
     */
    public abstract boolean isSpaceDifferentiable();

    /**
     * do we appear at the right of expression
     */
    public boolean appearAfter(Expression expression) {
        return this.appearanceOrder() > expression.appearanceOrder() ||
                this.appearanceOrder() == expression.appearanceOrder() && this.getDepth() > expression.getDepth();
    }

    /**
     * Coarse-grained ordering key used by {@link #appearAfter} to decide operand placement:
     * expressions with a higher {@code appearanceOrder()} are placed later (to the right).
     * The default, used by leaf-like nodes with no more specific rule, is {@code 0}; each
     * concrete node type overrides this with its own value (e.g. {@link BinaryExpression}
     * returns a higher value than {@link VariableExpression}).
     *
     * @return this node type's ordering key; higher values sort later
     */
    public int appearanceOrder() {
        return 0;
    }
    /**
     * How many levels of nesting separate this node from the tree's leaves; leaves are depth
     * {@code 0}. Set once at construction from the depth of this node's operand(s).
     */
    public int getDepth() {
        return depth;
    }
}