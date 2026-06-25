"""Helpers for building Dagster asset metadata (histograms, scatterplots, schemas).

Plots are rendered to base64 PNGs embedded in Markdown so they display inline on each
asset's materialization page in the Dagster UI.
"""

import base64
import functools
import io
from collections import defaultdict

import dagster as dg
import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # headless backend; render to PNG without a display
import matplotlib.pyplot as plt  # noqa: E402


def _graceful_plot(fn):
    """Wrap a plot helper so a rendering failure degrades to a text note instead of
    killing the asset/run. Any open figures are closed so a failure can't leak state."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - plotting must never fail the run
            plt.close("all")
            return dg.MetadataValue.text(f"{fn.__name__} failed: {type(exc).__name__}: {exc}")

    return wrapper


def _fig_to_md(fig) -> dg.MetadataValue:
    """Render a matplotlib figure to a base64 PNG embedded in Markdown."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode()
    return dg.MetadataValue.md(f"![plot](data:image/png;base64,{encoded})")


# @_graceful_plot
def histogram_md(values, *, discrete: bool = False, **kwargs) -> dg.MetadataValue:
    """A histogram of ``values`` as image metadata.

    ``kwargs`` are forwarded to ``Axes.set``.
    """
    fig, ax = plt.subplots()
    sns.histplot(list(values), discrete=discrete, edgecolor="black", ax=ax)
    ax.set(**{"ylabel": "count", **kwargs})
    return _fig_to_md(fig)


# @_graceful_plot
def scatter_md(x, y, **kwargs) -> dg.MetadataValue:
    """A scatterplot of (x, y) with a y=x reference line, as image metadata.

    ``kwargs`` are forwarded to ``Axes.set``.
    """
    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, alpha=0.6, ax=ax)
    combined = list(x) + list(y)
    lims = [min(combined), max(combined)]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set(**kwargs)
    ax.legend()
    return _fig_to_md(fig)


def table_schema_md(df: pd.DataFrame) -> dg.MetadataValue:
    """The DataFrame's column schema as table-schema metadata."""
    return dg.MetadataValue.table_schema(
        dg.TableSchema(
            columns=[dg.TableColumn(name=str(c), type=str(df[c].dtype)) for c in df.columns]
        )
    )


def metrics_table_md(metrics: dict[str, float]) -> dg.MetadataValue:
    """A two-column (metric, value) table from a name -> value mapping."""
    return dg.MetadataValue.table(
        records=[
            dg.TableRecord({"metric": name, "value": round(float(value), 4)})
            for name, value in metrics.items()
        ]
    )


def per_ligand_conformer_counts(batch) -> tuple[dict, dict]:
    """Return ``(total, converged)`` dicts mapping ligand id -> conformer count.

    ``converged`` is empty for batches that have not been optimised (no ``converged`` attr).
    """
    ids = batch.id
    converged = getattr(batch, "converged", None)
    total: dict = defaultdict(int)
    conv: dict = defaultdict(int)
    for idx in range(batch.n_conformers):
        lid = str(ids[idx])
        total[lid] += 1
        if converged is not None and bool(converged[idx]):
            conv[lid] += 1
    return dict(total), dict(conv)
