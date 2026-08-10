"""Publication style — matplotlib rcParams + colorblind-safe palette.

Palette: Okabe & Ito (2008), the current standard for colorblind-safe
qualitative colors in scientific plots. Sizes chosen to look clean at
column width in a two-column A&A / MNRAS layout.
"""

from __future__ import annotations

from typing import Iterable

# Okabe-Ito 8-color qualitative palette.
OKABE_ITO: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)


def apply_paper_style() -> None:
    """Set matplotlib rcParams for consistent publication output."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 11,
        "mathtext.fontset": "cm",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "pdf.fonttype": 42,     # embed TrueType (editable text in Illustrator/Inkscape)
        "ps.fonttype": 42,
        "axes.prop_cycle": cycler(color=list(OKABE_ITO)),
    })


def palette(n: int) -> tuple[str, ...]:
    """Return the first `n` Okabe-Ito colors, cycling if necessary.

    Includes black, which suits reference lines and thin marks. For filled
    areas (bars, large markers) use :func:`series_palette` instead.
    """
    return tuple(OKABE_ITO[i % len(OKABE_ITO)] for i in range(n))


# Okabe-Ito minus black. A large black fill reads as "missing data" or a
# printing error rather than as a category, so bar and patch series must not
# start at OKABE_ITO[0].
_FILL_SAFE: tuple[str, ...] = tuple(c for c in OKABE_ITO if c != "#000000")


def series_palette(n: int) -> tuple[str, ...]:
    """Colours for filled series (bars, patches, large markers).

    Same colourblind-safe hues as :func:`palette` with black removed.
    """
    return tuple(_FILL_SAFE[i % len(_FILL_SAFE)] for i in range(n))
