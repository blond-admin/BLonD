import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import sympy as sp
    from algebra_with_sympy import Eqn

    def step(text, *eqs, order=None):
        """Render a step description followed by one or more equations.

        order="none" prints products and sums in the order they were built (use with evaluate=False).
        """
        body = "\n\n".join(f"$$ {sp.latex(e, order=order)} $$" for e in eqs)
        return mo.md(f"{text}\n\n{body}")

    def lin(e, M=None, M_val=3):
        """Normal form for linearity checks: expand sums (M -> M_val), split integrals over terms, pull out constants."""
        if M is not None:
            e = e.replace(lambda z: isinstance(z, sp.Sum), lambda z: z.subs(M, M_val).doit(deep=False))
        if isinstance(e, sp.Integral):
            f = e.function
            for lim in e.limits:  # innermost first
                out = 0
                for term in sp.Add.make_args(sp.expand(lin(f))):
                    c, d = term.as_independent(lim[0], as_Add=False)
                    out += c * sp.Integral(d, lim)
                f = out
            return f
        if e.args and not isinstance(e, (sp.Indexed, sp.Symbol)):
            return e.func(*[lin(a) for a in e.args])
        return e

    def check(new, old, M=None):
        """SymPy check of one step: new - old in linearity normal form (0 means the step is correct)."""
        return Eqn(sp.Symbol(r"\text{difference}"), sp.expand(lin(new, M) - lin(old, M)))

    return Eqn, check, mo, sp, step


@app.cell
def _(mo):
    mo.md(r"""
    # Induced voltage from a wakefield

    - $t$, $\tau$: time within the bunch
    - $W(t)$: wake function (V/C)
    - $\lambda(\tau)$: line density, normalized to $\int \lambda(\tau)\,d\tau = 1$
    - $q$: particle charge, $N$: number of particles
    """)
    return


@app.cell
def _(sp):
    t, tau = sp.symbols("t tau", real=True)
    q, N = sp.symbols("q N", positive=True)
    W = sp.Function("W")
    lam = sp.Function("lambda")
    V_ind = sp.Function(r"V_{\mathrm{ind}}")
    return N, V_ind, W, lam, q, t, tau


@app.cell
def _(Eqn, N, V_ind, W, lam, q, sp, step, t, tau):
    eq1 = Eqn(V_ind(t), -q * N * sp.Integral(W(t - tau) * lam(tau), (tau, -sp.oo, sp.oo)))
    step(
        "**Step 1.** Induced voltage: convolution of the wake function with the line density.",
        eq1,
    )
    return (eq1,)


@app.cell
def _(Eqn, sp, step):
    x = sp.symbols("x", real=True)
    Box = sp.Function("Pi")
    half = sp.Rational(1, 2)
    box_def = sp.Piecewise((1, (x >= -half) & (x < half)), (0, True))
    eq2 = Eqn(Box(x), box_def)
    step(
        r"**Step 2.** Box function: 1 on the half-open interval $[-\tfrac12, \tfrac12)$, 0 elsewhere.",
        eq2,
    )
    return Box, eq2, x


@app.cell
def _(eq2, sp, step, t, x):
    t_k = sp.Symbol("t_k", real=True)
    dt = sp.Symbol(r"\Delta t", positive=True)
    eq3 = eq2.subs(x, (t - t_k) / dt)
    step(r"**Step 3.** Substitute $x = \frac{t - t_k}{\Delta t}$ on both sides.", eq3)
    return dt, eq3, t_k


@app.cell
def _(eq3, sp, step):
    cond4 = eq3.rhs.args[0].cond
    lo4 = next(r for r in cond4.args if isinstance(r, sp.GreaterThan))
    hi4 = next(r for r in cond4.args if isinstance(r, sp.StrictLessThan))
    step("**Step 4.** Take the two inequalities of the condition separately.", lo4, hi4)
    return hi4, lo4


@app.cell
def _(dt, hi4, lo4, step):
    lo5 = lo4.func(lo4.lhs * dt, lo4.rhs * dt)
    hi5 = hi4.func(hi4.lhs * dt, hi4.rhs * dt)
    step(
        r"**Step 5.** Multiply both sides by $\Delta t$ ($\Delta t > 0$, so the inequality directions stay).",
        lo5,
        hi5,
    )
    return hi5, lo5


@app.cell
def _(hi5, lo5, step, t_k):
    lo6 = lo5.func(lo5.lhs + t_k, lo5.rhs + t_k)
    hi6 = hi5.func(hi5.lhs + t_k, hi5.rhs + t_k)
    step(r"**Step 6.** Add $t_k$ to both sides.", lo6, hi6)
    return hi6, lo6


@app.cell
def _(Eqn, eq3, hi6, lo6, sp, step):
    eq7 = Eqn(eq3.lhs, sp.Piecewise((1, lo6 & hi6), (0, True)))
    step(
        r"**Step 7.** Put the new condition back into the box: 1 on the bin $[t_k - \tfrac{\Delta t}{2},\, t_k + \tfrac{\Delta t}{2})$.",
        eq7,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Voltage in one bin

    A discrete simulation keeps only one number $V_k$ per bin. The natural choice is
    the **average** of $V_{\mathrm{ind}}(t)$ over the bin.
    """)
    return


@app.cell
def _(Eqn, V_ind, dt, sp, step, t, t_k):
    V_k = sp.Symbol("V_k")
    eq8 = Eqn(V_k, sp.Integral(V_ind(t), (t, t_k - dt / 2, t_k + dt / 2)) / dt)
    step(
        r"**Step 8.** Bin voltage = average of $V_{\mathrm{ind}}$ over the bin: integrate over the bin, divide by its width.",
        eq8,
    )
    return (V_k,)


@app.cell
def _(Box, Eqn, V_ind, V_k, dt, eq2, sp, step, t, t_k, x):
    eq9 = Eqn(V_k, sp.Integral(Box((t - t_k) / dt) * V_ind(t), (t, -sp.oo, sp.oo)) / dt)
    # check: replace the box by its definition (Step 7) and let SymPy integrate
    chk9 = sp.Integral(eq2.rhs.subs(x, (t - t_k) / dt) * V_ind(t), (t, -sp.oo, sp.oo))
    step(
        r"**Step 9.** Extend the limits to $\pm\infty$ and multiply by the box (Step 7): "
        r"the box is 1 inside the bin and 0 outside, so the integral is unchanged."
        "\n\n*Check: with the box written out, SymPy gives back the integral of Step 8:*",
        eq9,
        Eqn(chk9, chk9.doit()),
    )
    return (eq9,)


@app.cell
def _(Box, Eqn, V_ind, dt, eq2, eq9, sp, step, t, t_k, x):
    eq10 = eq9.xreplace({Box((t - t_k) / dt): Box((t_k - t) / dt)})
    # check: the flipped box integrates over the same bin
    chk10 = sp.Integral(eq2.rhs.subs(x, (t_k - t) / dt) * V_ind(t), (t, -sp.oo, sp.oo))
    step(
        r"**Step 10.** The box is symmetric, $\Pi(-x) = \Pi(x)$ (except at the two edge points, "
        r"which do not change an integral). Flip its argument to $\frac{t_k - t}{\Delta t}$."
        "\n\n*Check: the flipped box also integrates over exactly the bin:*",
        eq10,
        Eqn(chk10, chk10.doit()),
    )
    return (eq10,)


@app.cell
def _(Box, Eqn, dt, sp, step):
    s = sp.Symbol("s", real=True)
    h = sp.Function("h")
    eq11 = Eqn(h(s), Box(s / dt) / dt)
    step(
        r"**Step 11.** Define the normalized box kernel $h$: a box of width $\Delta t$ and height $\frac{1}{\Delta t}$.",
        eq11,
    )
    return eq11, h, s


@app.cell
def _(Eqn, V_ind, V_k, eq10, eq11, h, s, sp, step, t, t_k):
    eq12 = Eqn(V_k, sp.Integral(h(t_k - t) * V_ind(t), (t, -sp.oo, sp.oo)))
    # check: putting the definition of h back in gives Step 10
    back12 = eq12.rhs.xreplace({h(t_k - t): eq11.rhs.subs(s, t_k - t)})
    step(
        r"**Step 12.** Use $h$ from Step 11 with $s = t_k - t$, and move $\frac{1}{\Delta t}$ into the integral. "
        r"This is the definition of a convolution evaluated at $t_k$: "
        r"$V_k = (h * V_{\mathrm{ind}})(t_k)$."
        "\n\n*Check: Step 12 minus Step 10, after putting $h$ back in:*",
        eq12,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(back12 - eq10.rhs)),
    )
    return (eq12,)


@app.cell
def _(Eqn, dt, eq2, s, sp, step, x):
    h_def = eq2.rhs.subs(x, s / dt) / dt
    area = sp.Integral(h_def, (s, -sp.oo, sp.oo))
    step(
        r"**Step 13.** $h$ has unit area, so it is an averaging kernel: $V_k$ is the "
        r"induced voltage smoothed by a moving average of width $\Delta t$, then sampled at $t_k$.",
        Eqn(area, area.doit()),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The voltage in one bin, restated
    """)
    return


@app.cell
def _(eq11, eq12, step):
    eq14 = eq12
    step(
        r"**Step 14.** Restate Step 12: the bin voltage is the induced voltage convolved with "
        r"the box kernel $h$ (Step 11), evaluated at the bin centre $t_k$.",
        eq14,
        eq11,
    )
    return (eq14,)


@app.cell
def _(V_ind, eq1, eq14, step, t):
    eq15 = eq14.xreplace({V_ind(t): eq1.rhs})
    step(r"**Step 15.** Insert Step 1 for $V_{\mathrm{ind}}(t)$.", eq15)
    return (eq15,)


@app.cell
def _(Eqn, N, V_k, W, check, eq15, h, lam, q, sp, step, t, t_k, tau):
    _inner = sp.Integral(W(t - tau) * lam(tau), (tau, -sp.oo, sp.oo))
    eq16 = Eqn(V_k, -q * N * sp.Integral(h(t_k - t) * _inner, (t, -sp.oo, sp.oo)))
    step(
        r"**Step 16.** Pull the constant $-qN$ out of the $t$-integral."
        "\n\n*Check: this step minus the previous one:*",
        eq16,
        check(eq16.rhs, eq15.rhs),
    )
    return (eq16,)


@app.cell
def _(Eqn, N, V_k, W, check, eq16, h, lam, q, sp, step, t, t_k, tau):
    eq17 = Eqn(
        V_k,
        -q * N * sp.Integral(
            sp.Integral(h(t_k - t) * W(t - tau) * lam(tau), (tau, -sp.oo, sp.oo)), (t, -sp.oo, sp.oo)
        ),
    )
    step(
        r"**Step 17.** Move $h(t_k - t)$ into the $\tau$-integral (it does not depend on $\tau$)."
        "\n\n*Check: this step minus the previous one:*",
        eq17,
        check(eq17.rhs, eq16.rhs),
    )
    return (eq17,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The line density is discrete too

    The simulation does not know $\lambda(\tau)$ as a function. It only has one number per bin:
    the histogram value $\lambda_j$ at the bin centre $t_j$. The integral in Step 17 needs
    $\lambda(\tau)$ at **every** $\tau$, so we have to build a continuous $\lambda(\tau)$ from the
    bin values: we interpolate.
    """)
    return


@app.cell
def _(Eqn, dt, sp, step):
    j = sp.Symbol("j", integer=True)
    M = sp.Symbol("M", integer=True, positive=True)
    t_0 = sp.Symbol("t_0", real=True)
    tg = sp.IndexedBase("{t}", real=True)  # grid points t_j; label "{t}" keeps it separate from the symbol t
    lam_j = sp.IndexedBase("lambda", real=True)
    eq18 = Eqn(tg[j], t_0 + j * dt)
    step(
        r"**Step 18.** Uniform grid of bin centres, $j = 0, \dots, M-1$ (the $t_k$ of the voltage is one of them).",
        eq18,
    )
    return M, eq18, j, lam_j, t_0, tg


@app.cell
def _(Eqn, j, lam, lam_j, step, tg):
    eq19 = Eqn(lam(tg[j]), lam_j[j])
    step(
        r"**Step 19.** Requirement: the continuous $\lambda(\tau)$ must pass through every bin value.",
        eq19,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The hat function $\Lambda = \Pi * \Pi$

    Linear interpolation between neighbouring bins uses the hat (triangle) function.
    We derive it as the convolution of two boxes.
    """)
    return


@app.cell
def _(Box, Eqn, sp, step, x):
    y = sp.Symbol("y", real=True)
    Lam = sp.Function(r"\Lambda")  # LaTeX name: "Lambda" would print with extra parentheses
    eq20 = Eqn(Lam(x), sp.Integral(Box(y) * Box(x - y), (y, -sp.oo, sp.oo)))
    step(r"**Step 20.** Define the hat $\Lambda$ as the convolution of two unit boxes.", eq20)
    return Lam, eq20, y


@app.cell
def _(Box, Eqn, Lam, sp, step, x, y):
    eq21 = Eqn(Lam(x), sp.Integral(Box(x - y), (y, -sp.Rational(1, 2), sp.Rational(1, 2))))
    step(
        r"**Step 21.** $\Pi(y)$ is 1 for $-\tfrac12 \le y < \tfrac12$ and 0 elsewhere (Step 2): "
        r"drop it and integrate only over that range.",
        eq21,
    )
    return


@app.cell
def _(eq2, sp, step, x, y):
    _cond = eq2.rhs.subs(x, x - y).args[0].cond
    lo22 = next(r for r in _cond.args if isinstance(r, sp.GreaterThan))
    hi22 = next(r for r in _cond.args if isinstance(r, sp.StrictLessThan))
    step(
        r"**Step 22.** Where is $\Pi(x - y) = 1$? Put $x - y$ into the condition of Step 2.",
        lo22,
        hi22,
    )
    return hi22, lo22


@app.cell
def _(hi22, lo22, step, x):
    lo23 = lo22.func(lo22.lhs - x, lo22.rhs - x)
    hi23 = hi22.func(hi22.lhs - x, hi22.rhs - x)
    step(r"**Step 23.** Subtract $x$ from both sides.", lo23, hi23)
    return hi23, lo23


@app.cell
def _(hi23, lo23, step):
    lo24 = lo23.reversedsign
    hi24 = hi23.reversedsign
    step(
        r"**Step 24.** Multiply both sides by $-1$; the inequality directions flip. "
        r"So the integrand of Step 21 is 1 for $x - \tfrac12 < y \le x + \tfrac12$, and "
        r"$\Lambda(x)$ is the **length of the overlap** of $[-\tfrac12, \tfrac12)$ and "
        r"$(x - \tfrac12, x + \tfrac12]$. The overlap depends on $x$: three cases.",
        lo24,
        hi24,
    )
    return


@app.cell
def _(Eqn, Lam, sp, step, x, y):
    eq25 = Eqn(Lam(x), sp.Integral(1, (y, x - sp.Rational(1, 2), sp.Rational(1, 2))))
    step(
        r"**Step 25. Case $0 \le x \le 1$.** Then $x - \tfrac12 \ge -\tfrac12$ and "
        r"$x + \tfrac12 \ge \tfrac12$: the overlap runs from $x - \tfrac12$ to $\tfrac12$.",
        eq25,
    )
    return (eq25,)


@app.cell
def _(Eqn, eq25, step):
    eq26 = Eqn(eq25.lhs, eq25.rhs.doit())
    step(r"**Step 26.** Evaluate the integral (case $0 \le x \le 1$).", eq26)
    return


@app.cell
def _(Eqn, Lam, sp, step, x, y):
    eq27 = Eqn(Lam(x), sp.Integral(1, (y, -sp.Rational(1, 2), x + sp.Rational(1, 2))))
    step(
        r"**Step 27. Case $-1 \le x \le 0$.** Then $x - \tfrac12 \le -\tfrac12$ and "
        r"$x + \tfrac12 \le \tfrac12$: the overlap runs from $-\tfrac12$ to $x + \tfrac12$.",
        eq27,
    )
    return (eq27,)


@app.cell
def _(Eqn, eq27, step):
    eq28 = Eqn(eq27.lhs, eq27.rhs.doit())
    step(r"**Step 28.** Evaluate the integral (case $-1 \le x \le 0$).", eq28)
    return


@app.cell
def _(Eqn, Lam, step, x):
    eq29 = Eqn(Lam(x), 0)
    step(
        r"**Step 29. Case $|x| > 1$.** For $x > 1$: $x - \tfrac12 > \tfrac12$. "
        r"For $x < -1$: $x + \tfrac12 < -\tfrac12$. Either way there is no overlap.",
        eq29,
    )
    return


@app.cell
def _(Eqn, Lam, eq2, sp, step, x, y):
    eq30 = Eqn(Lam(x), sp.Piecewise((1 - sp.Abs(x), sp.Abs(x) <= 1), (0, True)))
    # check: let SymPy convolve the two boxes directly and compare at test points
    _direct = sp.integrate(eq2.rhs.subs(x, y) * eq2.rhs.subs(x, x - y), (y, -sp.oo, sp.oo))
    _pts = [sp.Rational(k, 8) for k in range(-16, 17)]
    _ok = all(sp.simplify(_direct.subs(x, p) - eq30.rhs.subs(x, p)) == 0 for p in _pts)
    step(
        r"**Step 30.** Combine the cases: $1 - x$ (for $x \ge 0$) and $1 + x$ (for $x \le 0$) "
        r"are both $1 - |x|$."
        "\n\n*Check: SymPy convolves the two boxes directly (upper minus lower limit of the overlap). "
        f"It agrees with Step 30 at all {len(_pts)} test points " + r"$x = k/8$, $|k| \le 16$: " + f"**{_ok}**.*",
        eq30,
        Eqn(eq30.lhs, _direct),
    )
    return (eq30,)


@app.cell
def _(Eqn, Lam, eq30, sp, step, x):
    _area = sp.Integral(eq30.rhs, (x, -sp.oo, sp.oo))
    step(
        r"**Step 31.** Properties: 1 at its centre, 0 at the neighbouring integers "
        r"(and beyond, so $\Lambda(n) = 0$ for every integer $n \ne 0$), unit area.",
        Eqn(Lam(0), eq30.rhs.subs(x, 0)),
        Eqn(Lam(1), eq30.rhs.subs(x, 1)),
        Eqn(Lam(-1), eq30.rhs.subs(x, -1)),
        Eqn(_area, _area.doit()),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The hat of width $\Delta t$ from the kernel $h$
    """)
    return


@app.cell
def _(Eqn, h, s, sp, step):
    u = sp.Symbol("u", real=True)
    hh = sp.Function(r"\left(h * h\right)")
    eq32 = Eqn(hh(s), sp.Integral(h(u) * h(s - u), (u, -sp.oo, sp.oo)))
    step(r"**Step 32.** Convolution of the kernel $h$ (Step 11) with itself.", eq32)
    return eq32, hh, u


@app.cell
def _(eq11, eq32, h, s, step, u):
    eq33 = eq32.xreplace({h(u): eq11.rhs.subs(s, u), h(s - u): eq11.rhs.subs(s, s - u)})
    step(r"**Step 33.** Insert the definition of $h$ from Step 11.", eq33)
    return (eq33,)


@app.cell
def _(Box, Eqn, check, dt, eq33, hh, s, sp, step, u):
    eq34 = Eqn(hh(s), sp.Integral(Box(u / dt) * Box((s - u) / dt), (u, -sp.oo, sp.oo)) / dt**2)
    step(
        r"**Step 34.** Pull the constant $\frac{1}{\Delta t^2}$ out of the integral."
        "\n\n*Check: this step minus the previous one:*",
        eq34,
        check(eq34.rhs, eq33.rhs),
    )
    return (eq34,)


@app.cell
def _(Eqn, dt, eq34, hh, s, step, u, y):
    _I34 = eq34.rhs * dt**2
    eq35 = Eqn(hh(s), _I34.transform(u, (dt * y, y)) / dt**2)
    step(
        r"**Step 35.** Substitute $u = \Delta t\, y$, $du = \Delta t\, dy$ "
        r"($\Delta t > 0$, so the limits stay $\pm\infty$).",
        eq35,
    )
    return (eq35,)


@app.cell
def _(Box, Eqn, check, dt, eq35, hh, s, sp, step, y):
    eq36 = Eqn(hh(s), sp.Integral(Box(y) * Box(s / dt - y), (y, -sp.oo, sp.oo)) / dt)
    _norm = lambda e: e.replace(Box, lambda a: Box(sp.expand(a)))
    step(
        r"**Step 36.** Pull one $\Delta t$ out and write the argument as $\frac{s}{\Delta t} - y$."
        "\n\n*Check: this step minus the previous one:*",
        eq36,
        check(_norm(eq36.rhs), _norm(eq35.rhs)),
    )
    return (eq36,)


@app.cell
def _(Eqn, Lam, dt, eq20, eq36, hh, s, sp, step, x):
    eq37 = Eqn(hh(s), Lam(s / dt) / dt)
    step(
        r"**Step 37.** The integral is Step 20 with $x = \frac{s}{\Delta t}$:"
        "\n\n" + f"$$ {sp.latex(eq20.subs(x, s / dt))} $$"
        "\n\n*Check: Step 20 at that x minus the integral of Step 36:*",
        eq37,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(eq20.rhs.subs(x, s / dt) - eq36.rhs * dt)),
    )
    return (eq37,)


@app.cell
def _(dt, eq37, step):
    eq38 = (eq37 * dt).swap
    step(
        r"**Step 38.** Multiply both sides by $\Delta t$ and swap sides: "
        r"the hat of width $\Delta t$ is $\Delta t$ times the double box convolution.",
        eq38,
    )
    return (eq38,)


@app.cell
def _(mo):
    mo.md(r"""
    ### The interpolated line density
    """)
    return


@app.cell
def _(Eqn, Lam, M, dt, eq18, eq30, j, lam, lam_j, sp, step, tau, tg, x):
    eq39 = Eqn(lam(tau), sp.Sum(lam_j[j] * Lam((tau - tg[j]) / dt), (j, 0, M - 1)))
    # check (M = 3): evaluate at the middle grid point t_1
    _vals = (
        eq39.rhs.subs(M, 3)
        .doit()
        .subs({tg[i]: eq18.rhs.subs(j, i) for i in range(3)})
        .subs(tau, eq18.rhs.subs(j, 1))
        .replace(Lam, lambda a: eq30.rhs.subs(x, a))
    )
    step(
        r"**Step 39.** Linear interpolation: one hat per bin, of width $\Delta t$, centred on $t_j$. "
        r"At $\tau = t_m$ the hat argument is $m - j$, an integer, so by Step 31 only the term $j = m$ "
        r"survives: $\lambda(t_m) = \lambda_m$, as Step 19 requires."
        "\n\n*Check with $M = 3$ bins at $\\tau = t_1$:*",
        eq39,
        Eqn(lam(tg[1]), sp.simplify(_vals)),
    )
    return (eq39,)


@app.cell
def _(eq38, eq39, j, s, step, tau, tg):
    _hat = eq38.lhs.subs(s, tau - tg[j])
    eq40 = eq39.xreplace({_hat: eq38.rhs.subs(s, tau - tg[j])})
    step(r"**Step 40.** Replace each hat using Step 38 with $s = \tau - t_j$.", eq40)
    return (eq40,)


@app.cell
def _(Eqn, M, check, dt, eq40, hh, j, lam, lam_j, sp, step, tau, tg):
    eq41 = Eqn(lam(tau), dt * sp.Sum(lam_j[j] * hh(tau - tg[j]), (j, 0, M - 1)))
    step(
        r"**Step 41.** Pull the constant $\Delta t$ out of the sum."
        "\n\n*Check (sum written out for $M = 3$): this step minus the previous one:*",
        eq41,
        check(eq41.rhs, eq40.rhs, M),
    )
    return (eq41,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Three box convolutions in the discrete voltage
    """)
    return


@app.cell
def _(eq17, eq41, lam, step, tau):
    eq42 = eq17.xreplace({lam(tau): eq41.rhs})
    step(r"**Step 42.** Insert the interpolated line density (Step 41) into Step 17.", eq42)
    return (eq42,)


@app.cell
def _(
    Eqn,
    M,
    N,
    V_k,
    W,
    check,
    dt,
    eq42,
    h,
    hh,
    j,
    lam_j,
    q,
    sp,
    step,
    t,
    t_k,
    tau,
    tg,
):
    eq43 = Eqn(
        V_k,
        -q * N * dt * sp.Sum(
            lam_j[j]
            * sp.Integral(
                sp.Integral(h(t_k - t) * W(t - tau) * hh(tau - tg[j]), (tau, -sp.oo, sp.oo)),
                (t, -sp.oo, sp.oo),
            ),
            (j, 0, M - 1),
        ),
    )
    step(
        r"**Step 43.** Pull $\Delta t$ and the (finite) sum over $j$ out of both integrals: "
        r"integrals are linear."
        "\n\n*Check (sum written out for $M = 3$): this step minus the previous one:*",
        eq43,
        check(eq43.rhs, eq42.rhs, M),
    )
    return (eq43,)


@app.cell
def _(
    Eqn,
    M,
    N,
    V_k,
    W,
    check,
    dt,
    eq43,
    h,
    hh,
    j,
    lam_j,
    q,
    sp,
    step,
    t,
    t_k,
    tau,
    tg,
):
    eq44 = Eqn(
        V_k,
        -q * N * dt * sp.Sum(
            lam_j[j]
            * sp.Integral(
                h(t_k - t) * sp.Integral(W(t - tau) * hh(tau - tg[j]), (tau, -sp.oo, sp.oo)),
                (t, -sp.oo, sp.oo),
            ),
            (j, 0, M - 1),
        ),
    )
    step(
        r"**Step 44.** Move $h(t_k - t)$ back out of the $\tau$-integral (it does not depend on $\tau$)."
        "\n\n*Check (sum written out for $M = 3$): this step minus the previous one:*",
        eq44,
        check(eq44.rhs, eq43.rhs, M),
    )
    return (eq44,)


@app.cell
def _(Eqn, W, hh, j, sp, step, t, tau, tg):
    sig = sp.Symbol("sigma", real=True)
    _inner = sp.Integral(W(t - tau) * hh(tau - tg[j]), (tau, -sp.oo, sp.oo))
    eq45 = Eqn(_inner, _inner.transform(tau, (sig + tg[j], sig)))
    step(
        r"**Step 45.** Inner integral alone: substitute $\tau = \sigma + t_j$, $d\tau = d\sigma$.",
        eq45,
    )
    return eq45, sig


@app.cell
def _(Eqn, W, eq45, hh, j, sig, sp, step, t, tg):
    z = sp.Symbol("z", real=True)
    Whh = sp.Function(r"\left(W * h * h\right)")
    _def = Eqn(Whh(z), sp.Integral(W(z - sig) * hh(sig), (sig, -sp.oo, sp.oo)))
    eq46 = Eqn(eq45.rhs, Whh(t - tg[j]))
    step(
        r"**Step 46.** This is the convolution of $W$ with $h * h$, evaluated at $z = t - t_j$. "
        r"Definition:"
        "\n\n" + f"$$ {sp.latex(_def)} $$"
        "\n\n*Check: the definition at z = t - t_j minus the integral of Step 45:*",
        eq46,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(_def.rhs.subs(z, t - tg[j]) - eq45.rhs)),
    )
    return Whh, z


@app.cell
def _(Whh, eq44, eq45, j, step, t, tg):
    eq47 = eq44.xreplace({eq45.lhs: Whh(t - tg[j])})
    step(r"**Step 47.** Put Step 46 back into Step 44.", eq47)
    return (eq47,)


@app.cell
def _(Eqn, Whh, h, j, sp, step, t, t_k, tg):
    r = sp.Symbol("r", real=True)
    _outer = sp.Integral(h(t_k - t) * Whh(t - tg[j]), (t, -sp.oo, sp.oo))
    eq48 = Eqn(_outer, _outer.transform(t, (r + tg[j], r)))
    step(
        r"**Step 48.** Outer integral alone: substitute $t = r + t_j$, $dt = dr$.",
        eq48,
    )
    return eq48, r


@app.cell
def _(Eqn, Whh, eq48, h, j, r, sp, step, t_k, tg, z):
    hWhh = sp.Function(r"\left(h * W * h * h\right)")
    _def = Eqn(hWhh(z), sp.Integral(h(z - r) * Whh(r), (r, -sp.oo, sp.oo)))
    eq49 = Eqn(eq48.rhs, hWhh(t_k - tg[j]))
    step(
        r"**Step 49.** This is the convolution of $h$ with $W * h * h$, evaluated at $z = t_k - t_j$. "
        r"Definition:"
        "\n\n" + f"$$ {sp.latex(_def)} $$"
        "\n\n*Check: the definition at z = t_k - t_j minus the integral of Step 48:*",
        eq49,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(_def.rhs.subs(z, t_k - tg[j]) - eq48.rhs)),
    )
    return (hWhh,)


@app.cell
def _(eq47, eq48, hWhh, j, step, t_k, tg):
    eq50 = eq47.xreplace({eq48.lhs: hWhh(t_k - tg[j])})
    step(r"**Step 50.** Put Step 49 back into Step 47.", eq50)
    return


@app.cell
def _(Eqn, M, N, V_k, dt, j, lam_j, q, sp, step, t_k, tg):
    hhhW = sp.Function(r"\left(h * h * h * W\right)")
    eq51 = Eqn(V_k, -q * N * dt * sp.Sum(lam_j[j] * hhhW(t_k - tg[j]), (j, 0, M - 1)))
    step(
        r"**Step 51.** Convolution is commutative and associative: collect the three boxes in front. "
        r"One $h$ comes from averaging the voltage over the bin (Steps 8–13), "
        r"two come from the hat interpolation of the line density, $\Lambda = \Pi * \Pi$ (Steps 20–41).",
        eq51,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The pole-residue wake

    The wake function is known analytically as a sum of $P$ resonant modes (poles $s_p$, residues $r_p$).

    - $\alpha_p > 0$: decay rate of mode $p$
    - $\omega_p > 0$: oscillation frequency of mode $p$
    - $r_p$: complex residue of mode $p$; each pole comes with its complex conjugate, so $W$ is real
    - $\theta(t)$: Heaviside step (causality: no wake before the source passes), with $\theta(0) = \tfrac12$
    """)
    return


@app.cell
def _(Eqn, sp, step):
    p = sp.Symbol("p", integer=True)
    P = sp.Symbol("P", integer=True, positive=True)
    r_p = sp.IndexedBase("r")
    alpha = sp.IndexedBase("alpha", positive=True)
    omega = sp.IndexedBase("omega", positive=True)
    s_p = sp.IndexedBase("{s}")  # label "{s}" keeps it separate from the symbol s
    eq52 = Eqn(s_p[p], -alpha[p] + sp.I * omega[p])
    step(r"**Step 52.** Pole of mode $p$: decay rate as real part, oscillation frequency as imaginary part.", eq52)
    return P, alpha, eq52, omega, p, r_p, s_p


@app.cell
def _(Eqn, P, W, p, r_p, s_p, sp, step, t):
    # unevaluated products/sums keep the reading order: residue first, then the exponential
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _A = lambda *a: sp.Add(*a, evaluate=False)
    eq53 = Eqn(
        W(t),
        _M(
            sp.Heaviside(t),
            sp.Sum(
                _A(
                    _M(r_p[p], sp.exp(_M(s_p[p], t))),
                    _M(sp.conjugate(r_p[p]), sp.exp(_M(sp.conjugate(s_p[p]), t))),
                ),
                (p, 1, P),
            ),
        ),
    )
    step(
        r"**Step 53.** Pole-residue wake: each mode together with its complex-conjugate partner.",
        eq53,
        order="none",
    )
    return (eq53,)


@app.cell
def _(Eqn, P, W, alpha, eq52, eq53, omega, p, r_p, s_p, sp, step, t):
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _A = lambda *a: sp.Add(*a, evaluate=False)
    eq54 = Eqn(
        W(t),
        _M(
            sp.Heaviside(t),
            sp.Sum(
                _M(
                    sp.exp(_M(-alpha[p], t)),
                    _A(
                        _M(r_p[p], sp.exp(_M(sp.I, omega[p], t))),
                        _M(sp.conjugate(r_p[p]), sp.exp(_M(-sp.I, omega[p], t))),
                    ),
                ),
                (p, 1, P),
            ),
        ),
    )
    # check (P = 3): insert the poles of Step 52 into Step 53 and compare
    _d = (eq54.rhs.doit() - eq53.rhs.doit().xreplace({s_p[p]: eq52.rhs})).subs(P, 3).doit()
    step(
        r"**Step 54.** Insert the poles (Step 52) and split each exponential into a decaying factor "
        r"$e^{-\alpha_p t}$ and an oscillating factor $e^{\pm i\omega_p t}$."
        "\n\n*Check (sum written out for $P = 3$): this step minus Step 53 with the poles inserted:*",
        eq54,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(sp.powsimp(sp.expand(_d)))),
        order="none",
    )
    return (eq54,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. The pole-residue wake inside the triple box
    """)
    return


@app.cell
def _(Eqn, M, N, V_k, dt, j, lam_j, q, sp, step, t_k, tg, z):
    Wt = sp.Function(r"\tilde{W}")
    _hhhW = sp.Function(r"\left(h * h * h * W\right)")
    eq55 = Eqn(Wt(z), _hhhW(z))
    _V = Eqn(V_k, -q * N * dt * sp.Sum(lam_j[j] * Wt(t_k - tg[j]), (j, 0, M - 1)))
    step(
        r"**Step 55.** Name the kernel of Step 51 the effective wake $\tilde W$. "
        r"Step 51 then reads:",
        eq55,
        _V,
    )
    return (Wt,)


@app.cell
def _(Eqn, W, Wt, sp, step, t, z):
    hhh = sp.Function(r"\left(h * h * h\right)", real=True)
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    eq56 = Eqn(Wt(z), sp.Integral(_M(hhh(z - t), W(t)), (t, -sp.oo, sp.oo)))
    step(
        r"**Step 56.** Group the three boxes, $(h * h * h * W) = (h * h * h) * W$, and write out the "
        r"convolution with $W$: $\;(f * g)(z) = \int f(z - t)\, g(t)\, dt$.",
        eq56,
        order="none",
    )
    return eq56, hhh


@app.cell
def _(Eqn, W, Wt, eq54, eq56, hhh, sp, step, t, z):
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    eq57 = Eqn(Wt(z), sp.Integral(_M(hhh(z - t), eq54.rhs), (t, -sp.oo, sp.oo)))
    # check: same as replacing W(t) in Step 56 by Step 54
    _ev = lambda e: e.doit(integrals=False, sums=False)
    step(
        r"**Step 57.** Insert the pole-residue wake as written in Step 54."
        "\n\n*Check: this step minus Step 56 with W(t) replaced:*",
        eq57,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(_ev(eq57.rhs) - _ev(eq56.rhs.xreplace({W(t): eq54.rhs})))),
        order="none",
    )
    return (eq57,)


@app.cell
def _(Eqn, P, Wt, alpha, eq57, hhh, omega, p, r_p, sp, step, t, z):
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _A = lambda *a: sp.Add(*a, evaluate=False)
    _summand = _A(
        _M(r_p[p], sp.exp(_M(-alpha[p], t)), sp.exp(_M(sp.I, omega[p], t))),
        _M(sp.conjugate(r_p[p]), sp.exp(_M(-alpha[p], t)), sp.exp(_M(-sp.I, omega[p], t))),
    )
    eq58 = Eqn(Wt(z), sp.Integral(_M(hhh(z - t), sp.Heaviside(t), sp.Sum(_summand, (p, 1, P))), (t, -sp.oo, sp.oo)))
    # check: compare the summands (the term inside the sum over p)
    _old = eq57.rhs.atoms(sp.Sum).pop().function
    step(
        r"**Step 58.** Multiply the decaying factor $e^{-\alpha_p t}$ into the bracket."
        "\n\n*Check: summand of this step minus summand of the previous one:*",
        eq58,
        Eqn(sp.Symbol(r"\text{difference}"), sp.expand(_summand.doit() - _old.doit())),
        order="none",
    )
    return (eq58,)


@app.cell
def _(Eqn, P, Wt, alpha, eq58, hhh, omega, p, r_p, sp, step, t, z):
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _A = lambda *a: sp.Add(*a, evaluate=False)
    _summand = _A(
        _M(r_p[p], sp.exp(_M(_A(-alpha[p], _M(sp.I, omega[p])), t))),
        _M(sp.conjugate(r_p[p]), sp.exp(_M(_A(-alpha[p], -sp.I * omega[p]), t))),
    )
    eq59 = Eqn(Wt(z), sp.Integral(_M(hhh(z - t), sp.Heaviside(t), sp.Sum(_summand, (p, 1, P))), (t, -sp.oo, sp.oo)))
    _old = eq58.rhs.atoms(sp.Sum).pop().function
    step(
        r"**Step 59.** Combine the two exponentials of each term: $e^{a}\, e^{b} = e^{a + b}$."
        "\n\n*Check: summand of this step minus summand of the previous one:*",
        eq59,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(sp.powsimp(_summand.doit() - _old.doit()))),
        order="none",
    )
    return (eq59,)


@app.cell
def _(Eqn, P, Wt, eq52, eq59, hhh, p, r_p, s_p, sp, step, t, z):
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _A = lambda *a: sp.Add(*a, evaluate=False)
    _bar = Eqn(sp.conjugate(s_p[p]), sp.conjugate(eq52.rhs))
    _summand = _A(
        _M(r_p[p], sp.exp(_M(s_p[p], t))),
        _M(sp.conjugate(r_p[p]), sp.exp(_M(sp.conjugate(s_p[p]), t))),
    )
    eq60 = Eqn(Wt(z), sp.Integral(_M(hhh(z - t), sp.Heaviside(t), sp.Sum(_summand, (p, 1, P))), (t, -sp.oo, sp.oo)))
    # check: put the poles back in and compare with the summand of Step 59
    _old = eq59.rhs.atoms(sp.Sum).pop().function
    _new = _summand.doit().xreplace({s_p[p]: eq52.rhs})
    step(
        r"**Step 60.** The exponents are the pole of Step 52 and its complex conjugate "
        r"($\alpha_p$, $\omega_p$ are real):"
        "\n\n" + f"$$ {sp.latex(eq52)} \\qquad {sp.latex(_bar)} $$"
        "\n\nWrite the exponents as $s_p t$ and $\\bar s_p t$."
        "\n\n*Check: summand with the poles put back in, minus the summand of Step 59:*",
        eq60,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(sp.powsimp(_new - _old.doit()))),
        order="none",
    )
    return


@app.cell
def _(Eqn, P, Wt, hhh, p, r_p, s_p, sp, step, t, z):
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _A = lambda *a: sp.Add(*a, evaluate=False)
    eq61 = Eqn(
        Wt(z),
        sp.Integral(
            _M(
                hhh(z - t),
                sp.Sum(
                    _A(
                        _M(r_p[p], sp.exp(_M(s_p[p], t))),
                        _M(sp.conjugate(r_p[p]), sp.exp(_M(sp.conjugate(s_p[p]), t))),
                    ),
                    (p, 1, P),
                ),
            ),
            (t, 0, sp.oo),
        ),
    )
    step(
        r"**Step 61.** Causality (in Step 60): $\theta(t) = 0$ for $t < 0$ and $1$ for $t > 0$ "
        r"(the single point $t = 0$ does not change the integral). "
        r"Drop $\theta$ and start the integral at $0$.",
        eq61,
        order="none",
    )
    return (eq61,)


@app.cell
def _(Eqn, P, Wt, check, eq61, hhh, p, r_p, s_p, sp, step, t, z):
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _A = lambda *a: sp.Add(*a, evaluate=False)
    eq62 = Eqn(
        Wt(z),
        sp.Sum(
            _A(
                _M(r_p[p], sp.Integral(_M(hhh(z - t), sp.exp(_M(s_p[p], t))), (t, 0, sp.oo))),
                _M(
                    sp.conjugate(r_p[p]),
                    sp.Integral(_M(hhh(z - t), sp.exp(_M(sp.conjugate(s_p[p]), t))), (t, 0, sp.oo)),
                ),
            ),
            (p, 1, P),
        ),
    )
    _ev = lambda e: e.doit(integrals=False, sums=False)
    step(
        r"**Step 62.** Integrals are linear: pull the (finite) sum over $p$ and the residues out."
        "\n\n*Check (sum written out for $P = 3$): this step minus the previous one:*",
        eq62,
        check(_ev(eq62.rhs), _ev(eq61.rhs), P),
        order="none",
    )
    return


@app.cell
def _(Eqn, hhh, p, s_p, sp, step, t, z):
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _I = sp.Integral(_M(hhh(z - t), sp.exp(_M(s_p[p], t))), (t, 0, sp.oo))
    _Ibar = sp.Integral(_M(hhh(z - t), sp.exp(_M(sp.conjugate(s_p[p]), t))), (t, 0, sp.oo))
    eq63 = Eqn(_Ibar, sp.conjugate(_I))
    # check: the conjugate of the integrand is the integrand of the second integral
    _f = _I.function.doit(integrals=False)
    _g = _Ibar.function.doit(integrals=False)
    step(
        r"**Step 63.** The box kernel $h * h * h$, the limits and $t$ are real, so conjugating the "
        r"integral only conjugates $e^{s_p t}$ into $e^{\bar s_p t}$: the second integral is the "
        r"complex conjugate of the first."
        "\n\n*Check: conjugate of the first integrand minus the second integrand:*",
        eq63,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(sp.conjugate(_f) - _g)),
        order="none",
    )
    return


@app.cell
def _(Eqn, P, Wt, hhh, p, r_p, s_p, sp, step, t, z):
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _term = _M(r_p[p], sp.Integral(_M(hhh(z - t), sp.exp(_M(s_p[p], t))), (t, 0, sp.oo)))
    eq64 = Eqn(Wt(z), _M(2, sp.re(sp.Sum(_term, (p, 1, P)))))
    # check: w + conj(w) = 2 Re(w) for any complex w
    _w = sp.Symbol("w")
    step(
        r"**Step 64.** With Step 63, each mode is $w_p + \bar w_p = 2\,\mathrm{Re}\, w_p$ with "
        r"$w_p = r_p \int_0^\infty (h * h * h)(z - t)\, e^{s_p t}\, dt$; the real part of a sum is the sum of "
        r"the real parts. Effective wake, one complex integral per pole:"
        "\n\n*Check of the identity:*",
        eq64,
        Eqn(
            sp.Add(_w, sp.conjugate(_w), -2 * sp.re(_w), evaluate=False),
            sp.expand_complex(_w + sp.conjugate(_w) - 2 * sp.re(_w)),
        ),
        order="none",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. The discrete effective wake: near field and far field

    The effective wake is only ever needed at bin distances $z = t_k - t_j$.
    """)
    return


@app.cell
def _(Eqn, dt, eq18, j, sp, step, t_0, t_k, tg):
    k = sp.Symbol("k", integer=True)
    n = sp.Symbol("n", integer=True)
    _tk = Eqn(t_k, t_0 + k * dt)
    eq65 = Eqn(t_k - tg[j], sp.Mul(sp.Add(k, -j, evaluate=False), dt, evaluate=False))
    step(
        r"**Step 65.** $t_k$ and $t_j$ are both grid points (Step 18). Subtract them: $t_0$ cancels."
        "\n\n*Check: (t_0 + k Δt) - (t_0 + j Δt) - (k - j) Δt:*",
        _tk,
        eq18,
        eq65,
        Eqn(sp.Symbol(r"\text{difference}"), sp.expand(_tk.rhs - eq18.rhs - (k - j) * dt)),
        order="none",
    )
    return k, n


@app.cell
def _(Eqn, M, N, V_k, dt, j, k, lam_j, n, q, sp, step, z):
    Wn = sp.IndexedBase(r"\tilde{W}")
    _V = Eqn(
        V_k,
        sp.Mul(
            -q * N * dt,
            sp.Sum(
                sp.Mul(lam_j[j], Wn[sp.Add(k, -j, evaluate=False)], evaluate=False),
                (j, 0, sp.Add(M, -1, evaluate=False)),
            ),
            evaluate=False,
        ),
    )
    step(
        r"**Step 66.** Call the bin distance $n = k - j$, so $z = n\,\Delta t$, and write "
        r"$\tilde W_n = \tilde W(n\,\Delta t)$. The voltage of Step 55 becomes a discrete convolution:",
        Eqn(z, sp.Mul(n, dt, evaluate=False)),
        _V,
        order="none",
    )
    return (Wn,)


@app.cell
def _(mo):
    mo.md(r"""
    ### The triple box written out
    """)
    return


@app.cell
def _(Eqn, h, hh, hhh, sp, step, u):
    v = sp.Symbol("v", real=True)
    eq67 = Eqn(hhh(u), sp.Integral(sp.Mul(hh(v), h(u - v), evaluate=False), (v, -sp.oo, sp.oo)))
    step(
        r"**Step 67.** Group the triple box as $(h * h) * h$ and write out the convolution.",
        eq67,
        order="none",
    )
    return eq67, v


@app.cell
def _(eq11, eq37, eq67, h, hh, s, step, u, v):
    eq68 = eq67.xreplace({hh(v): eq37.rhs.subs(s, v), h(u - v): eq11.rhs.subs(s, u - v)})
    step(r"**Step 68.** Insert $h * h$ from Step 37 and $h$ from Step 11.", eq68)
    return (eq68,)


@app.cell
def _(Box, Eqn, Lam, check, dt, eq68, hhh, sp, step, u, v):
    eq69 = Eqn(hhh(u), sp.Integral(Lam(v / dt) * Box((u - v) / dt), (v, -sp.oo, sp.oo)) / dt**2)
    step(
        r"**Step 69.** Pull the constant $\frac{1}{\Delta t^2}$ out of the integral."
        "\n\n*Check: this step minus the previous one:*",
        eq69,
        check(eq69.rhs, eq68.rhs),
    )
    return (eq69,)


@app.cell
def _(Eqn, dt, eq69, hhh, step, u, v, y):
    eq70 = Eqn(hhh(u), (eq69.rhs * dt**2).transform(v, (dt * y, y)) / dt**2)
    step(
        r"**Step 70.** Substitute $v = \Delta t\, y$, $dv = \Delta t\, dy$ "
        r"($\Delta t > 0$, so the limits stay $\pm\infty$).",
        eq70,
    )
    return (eq70,)


@app.cell
def _(Box, Eqn, Lam, check, dt, eq70, hhh, sp, step, u, y):
    eq71 = Eqn(hhh(u), sp.Integral(Lam(y) * Box(u / dt - y), (y, -sp.oo, sp.oo)) / dt)
    _norm = lambda e: e.replace(Box, lambda a: Box(sp.expand(a))).replace(Lam, lambda a: Lam(sp.expand(a)))
    step(
        r"**Step 71.** Pull one $\Delta t$ out and write the box argument as $\frac{u}{\Delta t} - y$."
        "\n\n*Check: this step minus the previous one:*",
        eq71,
        check(_norm(eq71.rhs), _norm(eq70.rhs)),
    )
    return (eq71,)


@app.cell
def _(Box, Eqn, Lam, dt, eq71, hhh, sp, step, u, x, y):
    beta = sp.Function("beta")
    eq72 = Eqn(beta(x), sp.Integral(Lam(y) * Box(x - y), (y, -sp.oo, sp.oo)))
    _B = Eqn(hhh(u), beta(u / dt) / dt)
    step(
        r"**Step 72.** The integral is the hat convolved with one more box, $\beta = \Lambda * \Pi$, "
        r"evaluated at $x = \frac{u}{\Delta t}$. So the triple box is a scaled $\beta$:"
        "\n\n*Check: definition of β at x = u/Δt, divided by Δt, minus Step 71:*",
        eq72,
        _B,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(eq72.rhs.subs(x, u / dt) / dt - eq71.rhs)),
    )
    return (beta,)


@app.cell
def _(Eqn, Lam, beta, sp, step, x, y):
    _R = sp.Rational
    eq73 = Eqn(beta(x), sp.Integral(Lam(y), (y, x - _R(1, 2), x + _R(1, 2))))
    step(
        r"**Step 73.** By Step 24, $\Pi(x - y) = 1$ exactly for $x - \tfrac12 < y \le x + \tfrac12$ "
        r"and 0 elsewhere: $\beta$ is the hat averaged over a window of width 1 centred on $x$.",
        eq73,
    )
    return


@app.cell
def _(Eqn, Lam, beta, eq30, sp, step, x, y):
    _R = sp.Rational
    w = sp.Symbol("w", real=True)
    _I = sp.Integral(Lam(y), (y, -x - _R(1, 2), -x + _R(1, 2)))
    # substitution y = -w written out (SymPy's transform fails on this one): dy = -dw, limits swap
    _T = sp.Integral(Lam(-w), (w, x - _R(1, 2), x + _R(1, 2)))
    # check of the substitution with a test function f in place of Λ
    _f = lambda a: sp.exp(a) + a**3
    _sub = sp.simplify(
        sp.integrate(_f(y), (y, -x - _R(1, 2), -x + _R(1, 2))) - sp.integrate(_f(-w), (w, x - _R(1, 2), x + _R(1, 2)))
    )
    step(
        r"**Step 74.** $\beta$ is even: in $\beta(-x)$ substitute $y = -w$, $dy = -dw$; the window flips onto "
        r"$[x - \tfrac12, x + \tfrac12]$ (the minus sign swaps the limits back), and $\Lambda(-w) = \Lambda(w)$ "
        r"because $\Lambda$ depends on $|w|$ only (Step 30). So it is enough to compute $\beta(x)$ for $x \ge 0$."
        "\n\n*Check of the substitution with the test function $f(y) = e^y + y^3$ in place of Λ, "
        "and Λ(−w) − Λ(w) from Step 30:*",
        Eqn(beta(-x), _I),
        Eqn(_I, _T),
        Eqn(sp.Symbol(r"\text{difference}"), _sub),
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(eq30.rhs.subs(x, -w) - eq30.rhs.subs(x, w))),
    )
    return


@app.cell
def _(Eqn, beta, sp, step, x, y):
    _R = sp.Rational
    eq75 = Eqn(
        beta(x),
        sp.Add(
            sp.Integral(1 + y, (y, x - _R(1, 2), 0)),
            sp.Integral(1 - y, (y, 0, x + _R(1, 2))),
            evaluate=False,
        ),
    )
    step(
        r"**Step 75. Case $0 \le x \le \tfrac12$.** The window $[x - \tfrac12, x + \tfrac12]$ contains $0$ and "
        r"lies inside $[-1, 1]$. Split it at $0$: $\Lambda(y) = 1 + y$ for $y \le 0$ and $1 - y$ for $y \ge 0$.",
        eq75,
        order="none",
    )
    return (eq75,)


@app.cell
def _(Eqn, eq75, sp, step):
    eq76 = Eqn(eq75.lhs, sp.expand(eq75.rhs.doit()))
    step(r"**Step 76.** Evaluate (case $0 \le x \le \tfrac12$).", eq76)
    return


@app.cell
def _(Eqn, beta, sp, step, x, y):
    _R = sp.Rational
    eq77 = Eqn(beta(x), sp.Integral(1 - y, (y, x - _R(1, 2), 1)))
    step(
        r"**Step 77. Case $\tfrac12 \le x \le \tfrac32$.** The window starts in $[0, 1]$ and ends at or "
        r"beyond $1$, where $\Lambda$ is zero: integrate $1 - y$ up to $1$ only.",
        eq77,
    )
    return (eq77,)


@app.cell
def _(Eqn, eq77, sp, step, x):
    _R = sp.Rational
    eq78 = Eqn(eq77.lhs, sp.Mul(_R(1, 2), sp.Pow(sp.Add(_R(3, 2), -x, evaluate=False), 2), evaluate=False))
    step(
        r"**Step 78.** Evaluate (case $\tfrac12 \le x \le \tfrac32$)."
        "\n\n*Check: this result minus the evaluated integral of Step 77:*",
        eq78,
        Eqn(sp.Symbol(r"\text{difference}"), sp.expand(eq78.rhs.doit() - eq77.rhs.doit())),
        order="none",
    )
    return


@app.cell
def _(Eqn, beta, step, x):
    eq79 = Eqn(beta(x), 0)
    step(
        r"**Step 79. Case $x \ge \tfrac32$.** The window starts at or beyond $1$: $\Lambda$ is zero there.",
        eq79,
    )
    return


@app.cell
def _(Eqn, beta, eq2, eq30, sp, step, x, y):
    _R = sp.Rational
    eq80 = Eqn(
        beta(x),
        sp.Piecewise(
            (_R(3, 4) - x**2, sp.Abs(x) <= _R(1, 2)),
            (_R(1, 2) * (_R(3, 2) - sp.Abs(x)) ** 2, sp.Abs(x) <= _R(3, 2)),
            (0, True),
        ),
    )
    # check: SymPy convolves hat and box directly; compare at test points; area
    _hat = eq30.rhs.subs(x, y)
    _pts = [_R(k, 8) for k in range(-16, 17)]
    _ok = all(
        sp.simplify(sp.integrate(_hat * eq2.rhs.subs(x, p - y), (y, -2, 3)) - eq80.rhs.subs(x, p)) == 0 for p in _pts
    )
    _area = sp.Integral(eq80.rhs, (x, -2, 2))
    step(
        r"**Step 80.** Combine the cases, using $|x|$ for the even extension (Step 74). This is the "
        r"quadratic B-spline: zero for $|x| \ge \tfrac32$."
        "\n\n*Check: SymPy convolves Λ and Π directly; it agrees with Step 80 at all "
        + f"{len(_pts)}"
        + r" test points $x = k/8$, $|k| \le 16$: "
        + f"**{_ok}**. Area:*",
        eq80,
        Eqn(_area, _area.doit()),
    )
    return (eq80,)


@app.cell
def _(mo):
    mo.md(r"""
    ### One integral per pole, on the grid
    """)
    return


@app.cell
def _(Eqn, P, Wn, dt, hhh, n, p, r_p, s_p, sp, step, t):
    Ip = sp.Function("I_p")
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    eq81 = Eqn(
        Ip(n), sp.Integral(_M(hhh(sp.Add(n * dt, -t, evaluate=False)), sp.exp(_M(s_p[p], t))), (t, 0, sp.oo))
    )
    _W = Eqn(Wn[n], _M(2, sp.re(sp.Sum(_M(r_p[p], Ip(n)), (p, 1, P)))))
    step(
        r"**Step 81.** Step 64 at $z = n\,\Delta t$, with the integral of pole $p$ named $I_p(n)$:",
        _W,
        eq81,
        order="none",
    )
    return Ip, eq81


@app.cell
def _(Eqn, Ip, beta, dt, eq81, hhh, n, p, s_p, sp, step, t):
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    eq82 = Eqn(Ip(n), sp.Integral(_M(1 / dt, beta(n - t / dt), sp.exp(_M(s_p[p], t))), (t, 0, sp.oo)))
    # check: insert B(u) = beta(u/dt)/dt (Step 72) and compare
    _ins = eq81.rhs.replace(hhh, lambda a: beta(sp.expand(a / dt)) / dt)
    step(
        r"**Step 82.** Insert the triple box from Step 72 with $u = n\,\Delta t - t$, "
        r"so $\frac{u}{\Delta t} = n - \frac{t}{\Delta t}$."
        "\n\n*Check: this step minus Step 81 with the triple box replaced:*",
        eq82,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(eq82.rhs.doit(integrals=False) - _ins)),
        order="none",
    )
    return (eq82,)


@app.cell
def _(Eqn, Ip, dt, eq82, n, step, t, y):
    eq83 = Eqn(Ip(n), eq82.rhs.doit(integrals=False).transform(t, (dt * y, y)))
    step(
        r"**Step 83.** Substitute $t = \Delta t\, y$, $dt = \Delta t\, dy$: the factor $\frac{1}{\Delta t}$ "
        r"cancels, the limits stay $0$ and $\infty$.",
        eq83,
    )
    return (eq83,)


@app.cell
def _(Eqn, Ip, beta, dt, eq83, n, p, s_p, sp, step, y):
    sig_p = sp.IndexedBase(r"{\sigma}")
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    eq84 = Eqn(Ip(n), sp.Integral(_M(beta(n - y), sp.exp(_M(sig_p[p], y))), (y, 0, sp.oo)))
    step(
        r"**Step 84.** Name the dimensionless pole $\sigma_p = s_p\,\Delta t$ "
        r"(decay and phase per bin)."
        "\n\n*Check: Step 83 minus this step with σ_p = s_p Δt put back:*",
        Eqn(sig_p[p], _M(s_p[p], dt)),
        eq84,
        Eqn(
            sp.Symbol(r"\text{difference}"),
            sp.simplify(eq83.rhs - eq84.rhs.doit(integrals=False).xreplace({sig_p[p]: s_p[p] * dt})),
        ),
        order="none",
    )
    return (sig_p,)


@app.cell
def _(n, sp, step, y):
    _R = sp.Rational
    step(
        r"**Step 85.** By Step 80, $\beta(n - y) \ne 0$ only for $|n - y| < \tfrac32$, i.e. for "
        r"$n - \tfrac32 < y < n + \tfrac32$. The integral also needs $y \ge 0$. "
        r"Where this window sits relative to $y = 0$ gives three cases.",
        sp.StrictGreaterThan(y, n - _R(3, 2)),
        sp.StrictLessThan(y, n + _R(3, 2)),
    )
    return


@app.cell
def _(Eqn, Ip, n, step):
    step(
        r"**Step 86. Case $n \le -2$ (bins ahead).** The window ends at $n + \tfrac32 \le -\tfrac12 < 0$: "
        r"no overlap with $y \ge 0$. Causality, smeared by the boxes over one bin at most.",
        Eqn(Ip(n), 0),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Far field: $n \ge 2$
    """)
    return


@app.cell
def _(Eqn, Ip, beta, n, p, sig_p, sp, step, y):
    _R = sp.Rational
    eq87 = Eqn(Ip(n), sp.Integral(beta(n - y) * sp.exp(sig_p[p] * y), (y, n - _R(3, 2), n + _R(3, 2))))
    step(
        r"**Step 87. Case $n \ge 2$.** The window starts at $n - \tfrac32 \ge \tfrac12 > 0$: it lies "
        r"completely in $y > 0$, the cut at $y = 0$ (the jump of the wake) is never seen. "
        r"Integrate over the window.",
        eq87,
    )
    return (eq87,)


@app.cell
def _(Eqn, Ip, eq87, n, step, x, y):
    eq88 = Eqn(Ip(n), eq87.rhs.transform(y, (n - x, x)))
    step(
        r"**Step 88.** Substitute $y = n - x$, $dy = -dx$; the limits $n \mp \tfrac32$ become "
        r"$\pm\tfrac32$, and flipping them back absorbs the minus sign.",
        eq88,
    )
    return (eq88,)


@app.cell
def _(Eqn, Ip, beta, check, eq88, n, p, sig_p, sp, step, x):
    _R = sp.Rational
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    eq89 = Eqn(
        Ip(n),
        _M(
            sp.exp(_M(sig_p[p], n)),
            sp.Integral(_M(beta(x), sp.exp(_M(-sig_p[p], x))), (x, -_R(3, 2), _R(3, 2))),
        ),
    )
    step(
        r"**Step 89.** Split $e^{\sigma_p (n - x)} = e^{\sigma_p n}\, e^{-\sigma_p x}$ and pull "
        r"$e^{\sigma_p n}$ out (it does not depend on $x$)."
        "\n\n*Check: this step minus the previous one:*",
        eq89,
        check(eq89.rhs.doit(integrals=False), eq88.rhs),
        order="none",
    )
    return


@app.cell
def _(Eqn, p, sig_p, sp, step, x):
    _R = sp.Rational
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _e = sp.exp(_M(-sig_p[p], x))
    Fp = sp.Function("F")
    eq90 = Eqn(
        Fp(sig_p[p]),
        sp.Add(
            sp.Integral(_M(_R(1, 2), (_R(3, 2) + x) ** 2, _e), (x, -_R(3, 2), -_R(1, 2))),
            sp.Integral(_M(_R(3, 4) - x**2, _e), (x, -_R(1, 2), _R(1, 2))),
            sp.Integral(_M(_R(1, 2), (_R(3, 2) - x) ** 2, _e), (x, _R(1, 2), _R(3, 2))),
            evaluate=False,
        ),
    )
    step(
        r"**Step 90.** Name the remaining integral $F(\sigma_p)$ and split it along the pieces of "
        r"$\beta$ (Step 80; for $x < 0$, $|x| = -x$).",
        eq90,
        order="none",
    )
    return (eq90,)


@app.cell
def _(Eqn, eq90, p, sig_p, sp, step):
    _R = sp.Rational
    _sg = sig_p[p]
    _val = sum(sp.integrate(I.function, I.limits[0], conds="none") for I in eq90.rhs.args)
    _exps = sp.Add(
        sp.exp(_R(3, 2) * _sg), -3 * sp.exp(_R(1, 2) * _sg), 3 * sp.exp(-_R(1, 2) * _sg), -sp.exp(-_R(3, 2) * _sg),
        evaluate=False,
    )
    eq91 = Eqn(eq90.lhs, sp.Mul(_exps, sp.Pow(_sg, -3), evaluate=False))
    step(
        r"**Step 91.** Evaluate the three integrals and add them."
        "\n\n*Check: this result minus SymPy's evaluation of Step 90:*",
        eq91,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(sp.expand(eq91.rhs.doit() - _val))),
        order="none",
    )
    return (eq91,)


@app.cell
def _(Eqn, eq91, p, sig_p, sp, step, x):
    _R = sp.Rational
    _sg = sig_p[p]
    _cube = sp.Pow(sp.Add(sp.exp(_sg / 2), -sp.exp(-_sg / 2), evaluate=False), 3)
    _sinh = sp.Pow(sp.Mul(2, sp.sinh(_sg / 2), sp.Pow(_sg, -1), evaluate=False), 3, evaluate=False)
    eq92 = Eqn(eq91.lhs, _sinh)
    _box = sp.Integral(sp.exp(-_sg * x), (x, -_R(1, 2), _R(1, 2)))
    step(
        r"**Step 92.** The bracket is a cube, $(a - b)^3 = a^3 - 3a^2 b + 3ab^2 - b^3$ with "
        r"$a = e^{\sigma_p/2}$, $b = e^{-\sigma_p/2}$, and $a - b = 2\sinh\frac{\sigma_p}{2}$. "
        r"One factor per box: each box alone gives $\frac{2\sinh(\sigma_p/2)}{\sigma_p}$."
        "\n\n*Check: cube minus Step 91, and this result minus the single-box integral cubed:*",
        Eqn(eq91.lhs, sp.Mul(_cube, sp.Pow(_sg, -3), evaluate=False)),
        eq92,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(sp.expand(_cube / _sg**3 - eq91.rhs.doit()))),
        Eqn(
            sp.Symbol(r"\text{difference}"),
            sp.simplify((eq92.rhs.doit() - sp.integrate(_box.function, _box.limits[0], conds="none") ** 3).rewrite(sp.exp)),
        ),
        order="none",
    )
    return (eq92,)


@app.cell
def _(Eqn, Ip, eq92, n, p, sig_p, sp, step):
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    eq93 = Eqn(Ip(n), _M(sp.exp(_M(sig_p[p], n)), eq92.rhs))
    _ratio = sp.simplify(eq93.rhs.doit().subs(n, n + 1) / eq93.rhs.doit())
    step(
        r"**Step 93. Far field ($n \ge 2$).** Put $F$ back into Step 89. Each bin further away "
        r"multiplies by the same factor $e^{\sigma_p}$: a geometric sequence in $n$."
        "\n\n*Check: I_p(n+1) / I_p(n):*",
        eq93,
        Eqn(sp.Symbol(r"\frac{I_p(n+1)}{I_p(n)}"), _ratio),
        order="none",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Near field: $n = -1, 0, 1$

    For these the window $n - \tfrac32 < y < n + \tfrac32$ contains $y = 0$: the boxes overlap the
    jump of the wake. Integrate from $0$ to the end of the window, piece by piece of $\beta$.
    *Each result is checked against numerical integration of Step 84 at $\sigma_p = -0.3 + 2.1i$
    and against its limit $\sigma_p \to 0$.*
    """)
    return


@app.cell
def _(eq80, sp, x, y):
    import mpmath

    _bnum = sp.lambdify(x, eq80.rhs, "mpmath")
    _sv = sp.Rational(-3, 10) + sp.Rational(21, 10) * sp.I

    def near_check(n_val, closed, sg):
        """Closed form vs numerical integral of beta(n - y) e^{sigma y} over y >= 0, plus sigma -> 0 limit."""
        _num = mpmath.quad(
            lambda yy: _bnum(n_val - yy) * mpmath.exp(complex(_sv) * yy),
            [0, 0.5, 1, 1.5, 2, 2.5, 3],
        )
        _err = abs(complex(closed.subs(sg, _sv).evalf(30)) - complex(_num))
        _s0 = sp.Symbol("s0")  # sp.limit needs a plain symbol, not an indexed one
        _lim = sp.limit(closed.subs(sg, _s0), _s0, 0)
        _lim_direct = sp.integrate(eq80.rhs.subs(x, n_val - y), (y, 0, 3))
        return (
            f"*Check: |closed form − numerical integral| = {_err:.1e}; "
            f"limit σ→0: {_lim} (direct: {_lim_direct}).*"
        )

    return (near_check,)


@app.cell
def _(Eqn, Ip, p, sig_p, sp, step, y):
    _R = sp.Rational
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _e = sp.exp(_M(sig_p[p], y))
    eq94 = Eqn(Ip(-1), sp.Integral(_M(_R(1, 2), (_R(1, 2) - y) ** 2, _e), (y, 0, _R(1, 2))))
    step(
        r"**Step 94. Case $n = -1$.** Window up to $y = \tfrac12$. There $|{-1} - y| = 1 + y \in [1, \tfrac32]$: "
        r"outer piece of $\beta$, $\tfrac12(\tfrac32 - 1 - y)^2 = \tfrac12(\tfrac12 - y)^2$.",
        eq94,
        order="none",
    )
    return (eq94,)


@app.cell
def _(Eqn, eq94, p, sig_p, sp, step):
    _R = sp.Rational
    _sg = sig_p[p]
    _num = sp.Add(sp.exp(_sg / 2), -1, -_sg / 2, -_sg**2 / 8, evaluate=False)
    eq95 = Eqn(eq94.lhs, sp.Mul(_num, sp.Pow(_sg, -3), evaluate=False))
    _val = sp.integrate(eq94.rhs.function.doit(), eq94.rhs.limits[0], conds="none")
    step(
        r"**Step 95.** Evaluate ($n = -1$)."
        "\n\n*Check: this result minus SymPy's evaluation of Step 94:*",
        eq95,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(eq95.rhs.doit() - _val)),
        order="none",
    )
    return (eq95,)


@app.cell
def _(eq95, mo, near_check, p, sig_p):
    mo.md(near_check(-1, eq95.rhs.doit(), sig_p[p]))
    return


@app.cell
def _(Eqn, Ip, p, sig_p, sp, step, y):
    _R = sp.Rational
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _e = sp.exp(_M(sig_p[p], y))
    eq96 = Eqn(
        Ip(0),
        sp.Add(
            sp.Integral(_M(_R(3, 4) - y**2, _e), (y, 0, _R(1, 2))),
            sp.Integral(_M(_R(1, 2), (_R(3, 2) - y) ** 2, _e), (y, _R(1, 2), _R(3, 2))),
            evaluate=False,
        ),
    )
    step(
        r"**Step 96. Case $n = 0$.** Window up to $y = \tfrac32$, argument $|{-y}| = y$: "
        r"inner piece for $y \le \tfrac12$, outer piece for $\tfrac12 \le y \le \tfrac32$.",
        eq96,
        order="none",
    )
    return (eq96,)


@app.cell
def _(Eqn, eq96, p, sig_p, sp, step):
    _sg = sig_p[p]
    _num = sp.Add(sp.exp(3 * _sg / 2), -3 * sp.exp(_sg / 2), 2, -3 * _sg**2 / 4, evaluate=False)
    eq97 = Eqn(eq96.lhs, sp.Mul(_num, sp.Pow(_sg, -3), evaluate=False))
    _val = sum(sp.integrate(I.function.doit(), I.limits[0], conds="none") for I in eq96.rhs.args)
    step(
        r"**Step 97.** Evaluate ($n = 0$)."
        "\n\n*Check: this result minus SymPy's evaluation of Step 96:*",
        eq97,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(eq97.rhs.doit() - _val)),
        order="none",
    )
    return (eq97,)


@app.cell
def _(eq97, mo, near_check, p, sig_p):
    mo.md(near_check(0, eq97.rhs.doit(), sig_p[p]))
    return


@app.cell
def _(Eqn, Ip, p, sig_p, sp, step, y):
    _R = sp.Rational
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _e = sp.exp(_M(sig_p[p], y))
    eq98 = Eqn(
        Ip(1),
        sp.Add(
            sp.Integral(_M(_R(1, 2), (_R(1, 2) + y) ** 2, _e), (y, 0, _R(1, 2))),
            sp.Integral(_M(_R(3, 4) - (1 - y) ** 2, _e), (y, _R(1, 2), _R(3, 2))),
            sp.Integral(_M(_R(1, 2), (_R(5, 2) - y) ** 2, _e), (y, _R(3, 2), _R(5, 2))),
            evaluate=False,
        ),
    )
    step(
        r"**Step 98. Case $n = 1$.** Window up to $y = \tfrac52$, argument $1 - y$: "
        r"$y \in [0, \tfrac12]$: $|1 - y| \in [\tfrac12, 1]$, outer piece $\tfrac12(\tfrac12 + y)^2$; "
        r"$y \in [\tfrac12, \tfrac32]$: inner piece $\tfrac34 - (1 - y)^2$; "
        r"$y \in [\tfrac32, \tfrac52]$: $|1 - y| = y - 1$, outer piece $\tfrac12(\tfrac52 - y)^2$.",
        eq98,
        order="none",
    )
    return (eq98,)


@app.cell
def _(Eqn, eq98, p, sig_p, sp, step):
    _sg = sig_p[p]
    _num = sp.Add(
        sp.exp(5 * _sg / 2), -3 * sp.exp(3 * _sg / 2), 3 * sp.exp(_sg / 2), -1, _sg / 2, -_sg**2 / 8,
        evaluate=False,
    )
    eq99 = Eqn(eq98.lhs, sp.Mul(_num, sp.Pow(_sg, -3), evaluate=False))
    _val = sum(sp.integrate(I.function.doit(), I.limits[0], conds="none") for I in eq98.rhs.args)
    step(
        r"**Step 99.** Evaluate ($n = 1$)."
        "\n\n*Check: this result minus SymPy's evaluation of Step 98:*",
        eq99,
        Eqn(sp.Symbol(r"\text{difference}"), sp.simplify(eq99.rhs.doit() - _val)),
        order="none",
    )
    return (eq99,)


@app.cell
def _(eq99, mo, near_check, p, sig_p):
    mo.md(near_check(1, eq99.rhs.doit(), sig_p[p]))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Near field = far-field formula + a local correction
    """)
    return


@app.cell
def _(Eqn, eq91, n, p, sig_p, sp, step):
    _R = sp.Rational
    _sg = sig_p[p]
    Far = sp.Function(r"I^{\mathrm{far}}_p")
    _exps = sp.Add(
        sp.exp(_sg * (n + _R(3, 2))), -3 * sp.exp(_sg * (n + _R(1, 2))),
        3 * sp.exp(_sg * (n - _R(1, 2))), -sp.exp(_sg * (n - _R(3, 2))),
        evaluate=False,
    )
    eq100 = Eqn(Far(n), sp.Mul(_exps, sp.Pow(_sg, -3), evaluate=False))
    step(
        r"**Step 100.** Take the far-field formula (Steps 93 and 91) as a function for **every** $n$ and "
        r"multiply $e^{\sigma_p n}$ into the bracket."
        "\n\n*Check: this minus Step 93 written with Step 91:*",
        eq100,
        Eqn(
            sp.Symbol(r"\text{difference}"),
            sp.simplify(sp.expand(eq100.rhs.doit() - sp.exp(_sg * n) * eq91.rhs.doit())),
        ),
        order="none",
    )
    return Far, eq100


@app.cell
def _(Eqn, sp, step, x):
    Ecal = sp.Function(r"\mathcal{E}")
    eq101 = Eqn(Ecal(x), sp.Add(sp.exp(x), -1, -x, -x**2 / 2, evaluate=False))
    step(
        r"**Step 101.** Define $\mathcal{E}(x)$: the exponential minus its Taylor polynomial up to second "
        r"order. It is small near $x = 0$, $\mathcal{E}(x) \approx \frac{x^3}{6}$.",
        eq101,
        Eqn(Ecal(x), sp.series(sp.exp(x) - 1 - x - x**2 / 2, x, 0, 5)),
        order="none",
    )
    return Ecal, eq101


@app.cell
def _(Ecal, Eqn, eq100, eq101, eq95, eq97, eq99, n, p, sig_p, sp, step, x):
    _R = sp.Rational
    _sg = sig_p[p]
    Near = sp.Function(r"I^{\mathrm{near}}_p")
    _E = lambda a: Ecal(a * _sg)
    _forms = {
        -1: sp.Add(3 * _E(-_R(1, 2)), -3 * _E(-_R(3, 2)), _E(-_R(5, 2)), evaluate=False),
        0: sp.Add(-3 * _E(-_R(1, 2)), _E(-_R(3, 2)), evaluate=False),
        1: _E(-_R(1, 2)),
    }
    _closed = {-1: eq95.rhs, 0: eq97.rhs, 1: eq99.rhs}
    _eqs, _diffs = [], []
    for _n, _f in _forms.items():
        _eqs.append(Eqn(Near(_n), sp.Mul(_f, sp.Pow(_sg, -3), evaluate=False)))
        _val = (_f / _sg**3).doit().replace(Ecal, lambda a: eq101.rhs.doit().subs(x, a))
        _diffs.append(sp.simplify(sp.expand(_val - (_closed[_n].doit() - eq100.rhs.doit().subs(n, _n)))))
    step(
        r"**Step 102.** The near-field correction $I^{\mathrm{near}}_p(n) = I_p(n) - I^{\mathrm{far}}_p(n)$. "
        r"It only contains the exponentials with **negative** shift (the ones whose box part lies at $y < 0$, "
        r"cut off by causality), each minus its Taylor polynomial: local terms around the jump, "
        r"like derivatives of the wake at $t = 0^+$."
        "\n\n*Check: (Step 95, 97, 99) − (Step 100) − (this step), for n = −1, 0, 1:* "
        + ", ".join(f"**{d}**" for d in _diffs),
        *_eqs,
        order="none",
    )
    return (Near,)


@app.cell
def _(Eqn, Far, Ip, Near, P, Wn, n, p, r_p, sp, step):
    _M = lambda *f: sp.Mul(*f, evaluate=False)
    _A = lambda *a: sp.Add(*a, evaluate=False)
    eq103 = Eqn(Ip(n), _A(Far(n), Near(n)))
    _W = Eqn(Wn[n], _M(2, sp.re(sp.Sum(_M(r_p[p], _A(Far(n), Near(n))), (p, 1, P)))))
    step(
        r"**Step 103. The discrete effective wake.** For $n \ge -1$ every per-pole integral is the far-field "
        r"term plus the near-field correction; for $n \le -2$ it is zero (Step 86). "
        r"$I^{\mathrm{near}}_p(n) = 0$ for $n \ge 2$ (Step 93: there the far field is exact). "
        r"The far-field part is geometric in $n$ (factor $e^{\sigma_p}$ per bin), "
        r"the near-field part is non-zero on three bins only, $n = -1, 0, 1$.",
        eq103,
        _W,
        order="none",
    )
    return


if __name__ == "__main__":
    app.run()
