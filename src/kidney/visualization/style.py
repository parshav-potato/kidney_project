"""Shared plot utilities."""

import matplotlib.pyplot as plt


def save_and_show(fig, save_path: str | None = None) -> None:
    """Save figure if path given, then show."""
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def hide_top_right(ax) -> None:
    """Remove top and right spines from an axes."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
