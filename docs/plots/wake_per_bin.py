"""
Figures for the docstrings of `blond.physics.impedances.wake_per_bin`.

Each ``plot_<function name>`` draws the quantities defined in the docstring
of that function, with the same symbols, and is referenced from it with
``.. plot:: plots/wake_per_bin.py plot_<function name>``. The curves are
computed with the functions themselves. Times are in units of the bin
width ``DT``.
"""

import matplotlib.pyplot as plt
import mpmath
import numpy as np

from blond.physics.impedances.wake_per_bin.bin_average import (
    bspline_kink_kernel,
    bspline_step_kernel,
    causal_third_antiderivative_factor,
    quadratic_bspline,
    triple_box_average_linear_pole,
    triple_box_average_pole,
)

DT = 1.0  # bin width; every time axis is t / dt

EXACT_COLOR = "0.3"  # the function before averaging
AVERAGE_COLOR = "tab:blue"  # its bin average, what the code returns
INNER_COLOR = "tab:orange"  # first polynomial piece
OUTER_COLOR = "tab:green"  # second polynomial piece
SHADE = "0.93"


def _title(fig, function_name: str) -> None:
    fig.suptitle(function_name, family="monospace", fontsize=11)


def _mark_onset(ax, top: float) -> None:
    """Shade |t| <= 3/2 dt, the near branch, and label both branches."""
    ax.axvspan(-1.5, 1.5, color=SHADE, zorder=0)
    ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax.text(0.0, top, "near branch", ha="center", va="top", fontsize=8)
    ax.text(1.6, top, "far branch", ha="left", va="top", fontsize=8)


def plot_module() -> None:
    """Point taps vs averaged taps, and the alias suppression of B_2."""
    fig, (ax_taps, ax_alias) = plt.subplots(1, 2, figsize=(10, 3.6))
    _title(fig, "bin_average")

    # A pole whose oscillation is faster than the bins can resolve.
    pole, residue = complex(-0.6, 4.0), complex(1.0, 0.15)
    t = np.linspace(-3, 8, 2000)
    lags = np.arange(-3, 9.0)
    wake = np.where(t > 0, 2 * (residue * np.exp(pole * t)).real, 0.0)
    # At the jump, the point tap takes the mean W(0) = W(0+) / 2.
    point_taps = 2 * (residue * np.exp(pole * np.maximum(lags, 0))).real
    point_taps = np.where(lags > 0, point_taps, 0.5 * point_taps)
    point_taps[lags < 0] = 0.0
    average_taps = triple_box_average_pole(lags, pole, residue, DT)
    ax_taps.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax_taps.plot(t, wake, color=EXACT_COLOR, lw=0.8, label=r"$W(t)$")
    ax_taps.plot(
        t,
        triple_box_average_pole(t, pole, residue, DT),
        color=AVERAGE_COLOR,
        lw=1.5,
        label=r"$\overline{W}(t) = W * B_2$",
    )
    ax_taps.plot(
        lags,
        point_taps,
        "s",
        mfc="none",
        color="tab:red",
        label=r"point taps $W(k\Delta t)$",
    )
    ax_taps.plot(
        lags,
        average_taps,
        "o",
        color=AVERAGE_COLOR,
        label=r"averaged taps $\overline{W}(k\Delta t)$",
    )
    ax_taps.axvline(-1.5, color=AVERAGE_COLOR, ls=":", lw=1)
    ax_taps.annotate(
        "non-causal tap\n" r"($B_2$ reaches $-\frac{3}{2}\Delta t$)",
        xy=(-1, average_taps[2]),
        xytext=(-3.2, 1.2),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    ax_taps.set_xlabel(r"$t / \Delta t$")
    ax_taps.set_ylabel("wake")
    ax_taps.set_title(
        r"taps the solver applies, $p\,\Delta t = -0.6 + 4i$", fontsize=10
    )
    ax_taps.legend(fontsize=7, loc="upper right")

    # Weight a sampling scheme gives to the alias image j of a frequency.
    frequency = np.linspace(1e-3, 4.5, 2000)  # in units of 1 / dt
    tail = frequency > 0.5  # NOQA PLR2004
    ax_alias.semilogy(
        frequency,
        np.ones_like(frequency),
        color="tab:red",
        label="point sampling: 1",
    )
    ax_alias.semilogy(
        frequency,
        np.abs(np.sinc(frequency)) ** 3 + 1e-12,
        color=AVERAGE_COLOR,
        lw=1.5,
        label=r"$B_2$: $|\mathrm{sinc}(f\Delta t)|^3$",
    )
    ax_alias.semilogy(
        frequency[tail],
        (np.pi * frequency[tail]) ** -3,
        color="0.5",
        ls="--",
        lw=0.8,
        label=r"$\propto |j|^{-3}$",
    )
    for image in range(0, 5):
        if image % 2:
            ax_alias.axvspan(image - 0.5, image + 0.5, color=SHADE, zorder=0)
        ax_alias.text(
            max(image, 0.25),
            2.0,
            "band" if image == 0 else f"$j = {image}$",
            ha="center",
            fontsize=8,
        )
    ax_alias.set_ylim(1e-5, 5)
    ax_alias.set_xlim(0, 4.5)
    ax_alias.set_xlabel(
        r"$f\,\Delta t$  (image $j$ folds onto $f - j/\Delta t$)"
    )
    ax_alias.set_ylabel("weight of the image")
    ax_alias.set_title(r"weight given to frequency $f$", fontsize=10)
    ax_alias.legend(fontsize=7, loc="lower left")
    fig.tight_layout()


def plot_quadratic_bspline() -> None:
    """B_2 = box * box * box, with its two polynomial pieces."""
    x = np.linspace(-2, 2, 1601)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    _title(fig, "quadratic_bspline")
    box = np.where(np.abs(x) < 0.5, 1.0, 0.0)  # NOQA PLR2004
    ax.plot(x, box, color="0.75", lw=1, label="box")
    ax.plot(
        x,
        np.maximum(1 - np.abs(x), 0.0),
        color="0.55",
        lw=1,
        label="box * box",
    )
    bspline = np.array([quadratic_bspline(value) for value in x])
    inner = np.abs(x) <= 0.5  # NOQA PLR2004
    outer = (np.abs(x) > 0.5) & (np.abs(x) < 1.5)  # NOQA PLR2004
    ax.plot(
        x,
        np.where(inner, bspline, np.nan),
        color=INNER_COLOR,
        lw=2.5,
        label=r"$B_2 = \frac{3}{4} - x^2$,  $|x| \leq \frac{1}{2}$",
    )
    ax.plot(
        x,
        np.where(outer, bspline, np.nan),
        color=OUTER_COLOR,
        lw=2.5,
        label=r"$B_2 = \frac{1}{2}(\frac{3}{2} - |x|)^2$,  "
        r"$\frac{1}{2} < |x| < \frac{3}{2}$",
    )
    for edge in (-1.5, 1.5):
        ax.axvline(edge, color="0.5", ls=":", lw=1)
    ax.text(1.55, 0.6, "support\n" r"ends at $\pm\frac{3}{2}$", fontsize=8)
    ax.set_xlabel(r"$x$ (bin widths)")
    ax.set_ylabel(r"$B_2(x)$")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()


def _phi3_reference(pole_t: float) -> float:
    """Phi_3 at t = 1 in 50-digit arithmetic."""
    with mpmath.workdps(50):
        q = mpmath.mpf(pole_t)
        return float((mpmath.exp(q) - 1 - q - q**2 / 2) / q**3)


def plot_causal_third_antiderivative_factor() -> None:
    """Phi_3(t), and the cancellation that the series avoids."""
    fig, (ax_phi, ax_error) = plt.subplots(1, 2, figsize=(10, 3.4))
    _title(fig, "causal_third_antiderivative_factor")

    pole = complex(-0.5, 1.5)
    t = np.linspace(-1, 4, 1000)
    phi_3 = causal_third_antiderivative_factor(t, pole)
    series_end = 1.0 / abs(pole)
    ax_phi.axvspan(0, series_end, color=SHADE, zorder=0)
    ax_phi.text(
        series_end / 2, 2.6, "series\n" r"$|p t| < 1$", ha="center", fontsize=8
    )
    ax_phi.text(2.5, 2.9, "direct form", ha="center", fontsize=8)
    ax_phi.plot(
        t, phi_3.real, color=AVERAGE_COLOR, label=r"$\mathrm{Re}\,\varphi_3$"
    )
    ax_phi.plot(
        t,
        phi_3.imag,
        color=AVERAGE_COLOR,
        ls="--",
        label=r"$\mathrm{Im}\,\varphi_3$",
    )
    ax_phi.plot(
        t,
        np.where(t > 0, t**3 / 6, 0.0),
        color="0.6",
        ls=":",
        label=r"$t^3/6$ (leading term)",
    )
    ax_phi.set_ylim(-1, 3.5)
    ax_phi.set_xlabel(r"$t$ [s]")
    ax_phi.set_ylabel(r"$\varphi_3(t)$")
    ax_phi.set_title(
        rf"$p = {pole.real} + {pole.imag}i$ [rad/s]; zero for $t \leq 0$",
        fontsize=10,
    )
    ax_phi.legend(fontsize=7, loc="center left")

    magnitude = np.logspace(-6, 1, 300)
    pole_t = -magnitude  # a damped real pole, at t = 1
    reference = np.array([_phi3_reference(q) for q in pole_t])
    direct = (np.exp(pole_t) - 1 - pole_t - 0.5 * pole_t**2) / pole_t**3
    implemented = np.array(
        [
            causal_third_antiderivative_factor(np.array([1.0]), complex(q))[0]
            for q in pole_t
        ]
    ).real
    floor = 1e-17  # errors below it, exact results among them, sit on it
    ax_error.axvspan(1e-7, 1.0, color=SHADE, zorder=0)
    ax_error.axhline(2.2e-16, color="0.6", ls=":", lw=1)
    ax_error.text(12, 2.2e-16, "machine\nepsilon", va="center", fontsize=7)
    ax_error.loglog(
        magnitude,
        np.maximum(np.abs(direct / reference - 1), floor),
        color="tab:red",
        label="direct form everywhere",
    )
    ax_error.loglog(
        magnitude,
        np.maximum(np.abs(implemented / reference - 1), floor),
        color=AVERAGE_COLOR,
        label=r"this function (bottom edge: $< 10^{-17}$)",
    )
    ax_error.set_xlim(1e-6, 10)
    ax_error.text(3e-3, 1e1, "series used", ha="center", va="top", fontsize=8)
    ax_error.set_ylim(1e-17, 10)
    ax_error.set_xlabel(r"$|p\,t|$, real $p < 0$")
    ax_error.set_ylabel(r"relative error of $\varphi_3$")
    ax_error.set_title(
        r"the direct numerator cancels for small $|p t|$", fontsize=10
    )
    ax_error.legend(fontsize=7, loc="upper right")
    fig.tight_layout()


def plot_triple_box_average_pole() -> None:
    """W and its average for a complex, a real and a zero pole."""
    t = np.linspace(-3, 7, 2000)
    cases = (
        (
            complex(-0.4, 2.0),
            r"complex pole $p\,\Delta t = -0.4 + 2i$: $\kappa = 2$",
            (-1.6, 2.4),
        ),
        (
            complex(-0.5, 0.0),
            r"real pole $p\,\Delta t = -0.5$: $\kappa = 1$",
            (-0.15, 1.3),
        ),
        (
            complex(0.0, 0.0),
            r"zero pole: averaged step, $\kappa = 1$",
            (-0.15, 1.3),
        ),
    )
    residue = complex(1.0, 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    _title(fig, "triple_box_average_pole")
    for ax, (pole, title, ylim) in zip(axes, cases, strict=True):
        kappa = 2.0 if pole.imag else 1.0
        wake = np.where(t > 0, kappa * (residue * np.exp(pole * t)).real, 0.0)
        ax.set_ylim(*ylim)
        _mark_onset(ax, ylim[1])
        ax.plot(
            t,
            wake,
            color=EXACT_COLOR,
            lw=0.8,
            label=r"$W = \kappa\,\mathrm{Re}[\rho e^{p t}]\,\Theta(t)$",
        )
        ax.plot(
            t,
            triple_box_average_pole(t, pole, residue, DT),
            color=AVERAGE_COLOR,
            lw=2,
            label=r"$\overline{W}$ (returned)",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$t / \Delta t$")
    axes[0].set_ylabel(r"wake, $\rho = 1$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))


def plot_triple_box_average_linear_pole() -> None:
    """Re[(c_0 + c_1 t) e^{p t}] Theta(t) and its average."""
    t = np.linspace(-3, 14, 2000)
    pole = complex(-0.25, 1.2)
    constant, slope = complex(1.0), complex(0.6)
    wake = np.where(
        t > 0, ((constant + slope * t) * np.exp(pole * t)).real, 0.0
    )
    envelope = np.where(
        t > 0, (constant + slope * t).real * np.exp(pole.real * t), np.nan
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    _title(fig, "triple_box_average_linear_pole")
    ax.set_ylim(-2.3, 2.4)
    _mark_onset(ax, 2.35)
    ax.text(
        0,
        -2.2,
        "third differences\n" r"of $\varphi_3$ and $\psi_3$",
        ha="center",
        fontsize=7,
    )
    ax.text(
        7.5,
        -2.2,
        r"$[(c_0 + c_1 t) E^3 + 3 c_1 \Delta t E^2 G]"
        r"\,e^{p (t - 3\Delta t/2)}$",
        ha="center",
        fontsize=8,
    )
    ax.plot(
        t,
        envelope,
        color="0.6",
        ls=":",
        lw=1,
        label=r"envelope $(c_0 + c_1 t)\,e^{\mathrm{Re}(p)\,t}$",
    )
    ax.plot(
        t,
        wake,
        color=EXACT_COLOR,
        lw=0.8,
        label=r"$\mathrm{Re}[(c_0 + c_1 t)\,e^{p t}]\,\Theta(t)$",
    )
    ax.plot(
        t,
        triple_box_average_linear_pole(t, pole, constant, slope, DT),
        color=AVERAGE_COLOR,
        lw=2,
        label="bin average (returned)",
    )
    ax.set_xlabel(r"$t / \Delta t$")
    ax.set_ylabel("wake")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()


def _plot_edge_kernel(function_name, kernel, edge, labels, legend_loc) -> None:
    """Edge vs its B_2 average (left) and their difference (right)."""
    edge_label, average_label, inner_label, outer_label = labels
    u = np.linspace(-2.2, 2.2, 1761)
    excess = kernel(u)
    fig, (ax_edge, ax_excess) = plt.subplots(1, 2, figsize=(10, 3.2))
    _title(fig, function_name)
    for ax in (ax_edge, ax_excess):
        ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
        for boundary in (-1.5, 1.5):
            ax.axvline(boundary, color="0.5", ls=":", lw=1)
        ax.set_xlabel(r"$u$ (bin widths)")
    ax_edge.plot(u, edge(u), color=EXACT_COLOR, lw=1, label=edge_label)
    ax_edge.plot(
        u, edge(u) + excess, color=AVERAGE_COLOR, lw=2, label=average_label
    )
    ax_edge.set_title("before and after averaging", fontsize=10)
    ax_edge.legend(fontsize=8, loc="upper left")
    inner = np.abs(u) <= 0.5  # NOQA PLR2004
    support = np.abs(u) < 1.5  # NOQA PLR2004
    outer = (np.abs(u) > 0.5) & support  # NOQA PLR2004
    # Each side of u = 0 separately, so a jump there is not drawn.
    for side, label_suffix in ((u < 0, ""), (u >= 0, "_")):
        ax_excess.plot(
            u,
            np.where(inner & side, excess, np.nan),
            color=INNER_COLOR,
            lw=2.5,
            label=label_suffix + inner_label,
        )
        ax_excess.plot(
            u,
            np.where(outer & side, excess, np.nan),
            color=OUTER_COLOR,
            lw=2.5,
            label=label_suffix + outer_label,
        )
    ax_excess.plot(u, np.where(support, np.nan, excess), color="0.6", lw=1.5)
    ax_excess.set_title(
        r"returned: the difference, zero for $|u| \geq \frac{3}{2}$",
        fontsize=10,
    )
    if kernel is bspline_step_kernel:
        ax_excess.plot(0.0, kernel(np.array([0.0]))[0], "o", color="k")
        ax_excess.annotate(
            r"$u = 0$: $-\frac{1}{2}$ ($\Theta(0) = 1$)",
            xy=(0.0, -0.5),
            xytext=(0.35, -0.42),
            fontsize=8,
        )
        ax_excess.text(
            1.9, 0.35, r"upper sign: $u \geq 0$", ha="right", fontsize=8
        )
    ax_excess.legend(fontsize=8, loc=legend_loc)
    fig.tight_layout()


def plot_bspline_kink_kernel() -> None:
    """R(u) - (u)_+ and its two polynomial pieces."""
    _plot_edge_kernel(
        "bspline_kink_kernel",
        bspline_kink_kernel,
        lambda u: np.maximum(u, 0.0),
        (
            r"kink $(u)_+$",
            r"average $R(u)$",
            r"$\frac{13}{64} - \frac{|u|}{2} + \frac{3u^2}{8}"
            r" - \frac{u^4}{12}$",
            r"$\frac{(3/2 - |u|)^4}{24}$",
        ),
        "upper left",
    )


def plot_bspline_step_kernel() -> None:
    """C(u) - Theta(u), odd about the step."""
    _plot_edge_kernel(
        "bspline_step_kernel",
        bspline_step_kernel,
        lambda u: np.where(u >= 0, 1.0, 0.0),
        (
            r"step $\Theta(u)$",
            r"average $C(u)$",
            r"$\mp(\frac{1}{2} - \frac{3|u|}{4} + \frac{|u|^3}{3})$",
            r"$\mp\frac{(3/2 - |u|)^3}{6}$",
        ),
        "lower left",
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    for name, function in list(globals().items()):
        if name.startswith("plot_"):
            function()
            plt.savefig(f"{out_dir}/{name}.png", dpi=90)
            plt.close("all")
