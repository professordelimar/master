# -*- coding: utf-8 -*-
"""
plotting.py
===========

High-quality plotting utilities for gravimetric, magnetic and geophysical maps.

Author: Nelson Ribeiro Filho
Revised/organized by: ChatGPT

Main features
-------------
- Simple line plots, scatter plots, histograms and contour maps.
- Regular-grid contourf/contour.
- Irregular-data tricontourf/tricontour.
- Drawing utilities: rectangle, circle and polygon.
- Cartopy geographic basemap from longitude/latitude limits.
- Geographic scatter and tricontourf maps.
- Standardized figure saving with dpi=300, bbox_inches="tight" and transparent background.
- Default colormap: "coolwarm".
- Optional Crameri colormaps: "vik", "roma", "batlow", if cmcrameri is installed.

Important design decision
-------------------------
This file intentionally does not keep the old Basemap-based structure. Basemap is deprecated
for many modern workflows. Cartopy is used for geographic maps.

Coordinate convention
---------------------
Cartesian maps:
    x, y = generic coordinates.

Geographic maps:
    lon, lat = longitude and latitude in decimal degrees.
    area = [lon_min, lon_max, lat_min, lat_max].
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.ticker import MultipleLocator


ArrayLike = Union[np.ndarray, Sequence[float]]
AreaLike = Sequence[float]


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_CMAP = "coolwarm"
DEFAULT_FIGSIZE = (8.0, 6.0)
DEFAULT_DPI = 300


# ============================================================
# BASIC INTERNAL UTILITIES
# ============================================================

def _to_array(a: ArrayLike) -> np.ndarray:
    """
    Convert input to numpy array.
    """
    return np.asarray(a)


def _finite_data(data: ArrayLike) -> np.ndarray:
    """
    Return finite data as a flattened array.
    """
    arr = np.asarray(data)
    return arr[np.isfinite(arr)]


def _check_same_shape(*arrays: ArrayLike) -> Tuple[np.ndarray, ...]:
    """
    Check if all arrays have the same shape and return them as numpy arrays.
    """
    arrs = tuple(np.asarray(a) for a in arrays)
    shape0 = arrs[0].shape

    for arr in arrs[1:]:
        if arr.shape != shape0:
            raise ValueError("All input arrays must have the same shape!")

    return arrs


def _get_fig_ax(
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    projection=None,
) -> Tuple[Figure, Axes]:
    """
    Return figure and axis. If ax is None, create a new one.
    """
    if ax is not None:
        return ax.figure, ax

    if projection is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=projection)

    return fig, ax


def _format_axis(
    ax: Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    grid: bool = True,
    equal_aspect: bool = False,
    labelsize: int = 11,
    titlesize: int = 13,
) -> None:
    """
    Standard axis formatting.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize)

    if title is not None:
        ax.set_title(title, fontsize=titlesize)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if grid is True:
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    if equal_aspect is True:
        ax.set_aspect("equal", adjustable="box")

    ax.tick_params(labelsize=9)


def _reshape_grid(
    x: ArrayLike,
    y: ArrayLike,
    data: ArrayLike,
    shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare regular-grid arrays.

    If shape is provided, x, y and data are reshaped to that shape.
    If shape is None, x, y and data are used as provided.
    """
    x, y, data = _check_same_shape(x, y, data)

    if shape is not None:
        return x.reshape(shape), y.reshape(shape), data.reshape(shape)

    if x.ndim != 2 or y.ndim != 2 or data.ndim != 2:
        raise ValueError(
            "For regular contour maps, x, y and data must be 2D arrays, "
            "or shape must be provided to reshape 1D arrays."
        )

    return x, y, data


def my_get_colormap(colormap: Union[str, object] = DEFAULT_CMAP):
    """
    Return a Matplotlib-compatible colormap.

    Recommended options
    -------------------
    - "coolwarm"
    - "RdYlBu_r"
    - "vik"
    - "roma"
    - "batlow"

    Crameri colormaps require:
        pip install cmcrameri
    or:
        conda install -c conda-forge cmcrameri
    """
    if not isinstance(colormap, str):
        return colormap

    crameri_names = {
        "vik", "vik_r",
        "roma", "roma_r",
        "batlow", "batlow_r",
        "hawaii", "hawaii_r",
        "lajolla", "lajolla_r",
        "devon", "devon_r",
        "berlin", "berlin_r",
    }

    if colormap in crameri_names:
        try:
            import cmcrameri.cm as cmc
            return getattr(cmc, colormap)
        except Exception:
            print(
                f"Warning: Crameri colormap '{colormap}' is unavailable. "
                f"Using '{DEFAULT_CMAP}'."
            )
            return plt.get_cmap(DEFAULT_CMAP)

    return plt.get_cmap(colormap)


def my_get_norm(
    data: Optional[ArrayLike] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = False,
):
    """
    Return a Matplotlib normalization object.

    If center_zero=True, a symmetric TwoSlopeNorm centered at zero is used.
    This is useful for gravity/magnetic residuals and derivative maps.
    """
    if data is not None:
        arr = np.asarray(data)
        if vmin is None:
            vmin = float(np.nanmin(arr))
        if vmax is None:
            vmax = float(np.nanmax(arr))

    if center_zero is True:
        if vmin is None or vmax is None:
            raise ValueError("vmin and vmax must be defined when center_zero=True.")
        amplitude = max(abs(vmin), abs(vmax))
        return TwoSlopeNorm(vmin=-amplitude, vcenter=0.0, vmax=amplitude)

    return Normalize(vmin=vmin, vmax=vmax)


def my_add_colorbar(
    mappable,
    ax: Optional[Axes] = None,
    label: Optional[str] = None,
    orientation: str = "vertical",
    shrink: float = 0.85,
    pad: float = 0.04,
    aspect: float = 30.0,
    ticks: Optional[Sequence[float]] = None,
    labelsize: int = 11,
    ticksize: int = 9,
):
    """
    Add a colorbar to a plot.
    """
    if ax is None:
        ax = plt.gca()

    cbar = plt.colorbar(
        mappable,
        ax=ax,
        orientation=orientation,
        shrink=shrink,
        pad=pad,
        aspect=aspect,
    )

    if label is not None:
        cbar.set_label(label, fontsize=labelsize)

    if ticks is not None:
        cbar.set_ticks(ticks)

    cbar.ax.tick_params(labelsize=ticksize)

    return cbar


# ============================================================
# DRAWING FUNCTIONS
# ============================================================

def my_draw_rectangle(
    area: AreaLike,
    ax: Optional[Axes] = None,
    edgecolor: str = "black",
    facecolor: str = "none",
    linewidth: float = 1.5,
    linestyle: str = "-",
    alpha: float = 1.0,
    label: Optional[str] = None,
    xy2ne: bool = False,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
):
    """
    Draw a rectangle.

    Parameters
    ----------
    area : list
        [x_min, x_max, y_min, y_max].
    xy2ne : bool
        If True, swaps x/y order.
    """
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    x1, x2, y1, y2 = area

    if xy2ne is True:
        x1, x2, y1, y2 = y1, y2, x1, x2

    patch = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
    )

    ax.add_patch(patch)

    if label is not None:
        ax.legend()

    return patch


def my_draw_circle(
    center: Tuple[float, float],
    radius: float,
    ax: Optional[Axes] = None,
    edgecolor: str = "black",
    facecolor: str = "none",
    linewidth: float = 1.5,
    linestyle: str = "-",
    alpha: float = 1.0,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
):
    """
    Draw a circle.
    """
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    patch = Circle(
        center,
        radius,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
    )

    ax.add_patch(patch)

    if label is not None:
        ax.legend()

    return patch


def my_draw_polygon(
    vertices: ArrayLike,
    ax: Optional[Axes] = None,
    edgecolor: str = "black",
    facecolor: str = "none",
    linewidth: float = 1.5,
    linestyle: str = "-",
    alpha: float = 1.0,
    label: Optional[str] = None,
    closed: bool = True,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
):
    """
    Draw a polygon from vertices with shape (n_vertices, 2).
    """
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    vertices = np.asarray(vertices)

    patch = Polygon(
        vertices,
        closed=closed,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
    )

    ax.add_patch(patch)

    if label is not None:
        ax.legend()

    return patch


# ============================================================
# BASIC PLOTS
# ============================================================

def my_plot(
    x: ArrayLike,
    y: ArrayLike,
    ax: Optional[Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    color: Optional[str] = None,
    linestyle: str = "-",
    linewidth: float = 1.5,
    marker: Optional[str] = None,
    markersize: float = 4.0,
    label: Optional[str] = None,
    grid: bool = True,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Make a simple line plot.
    """
    x, y = _check_same_shape(x, y)
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    line, = ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
        label=label,
    )

    _format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title,
                 xlim=xlim, ylim=ylim, grid=grid)

    if label is not None:
        ax.legend()

    return fig, ax, line


def my_scatter(
    x: ArrayLike,
    y: ArrayLike,
    values: Optional[ArrayLike] = None,
    ax: Optional[Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    colormap: Union[str, object] = DEFAULT_CMAP,
    color: str = "black",
    size: float = 20.0,
    marker: str = "o",
    alpha: float = 1.0,
    edgecolors: str = "none",
    colorbar: bool = False,
    colorbar_label: Optional[str] = None,
    colorbar_orientation: str = "vertical",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = False,
    grid: bool = True,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Make a scatter plot.

    If values is provided, values control point colors.
    """
    x, y = _check_same_shape(x, y)
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    if values is None:
        sc = ax.scatter(
            x,
            y,
            s=size,
            color=color,
            marker=marker,
            alpha=alpha,
            edgecolors=edgecolors,
        )
    else:
        values = np.asarray(values)
        if values.shape != x.shape:
            raise ValueError("values must have the same shape as x and y.")

        cmap = my_get_colormap(colormap)
        norm = my_get_norm(values, vmin=vmin, vmax=vmax, center_zero=center_zero)

        sc = ax.scatter(
            x,
            y,
            c=values,
            s=size,
            cmap=cmap,
            norm=norm,
            marker=marker,
            alpha=alpha,
            edgecolors=edgecolors,
        )

        if colorbar is True:
            my_add_colorbar(
                sc,
                ax=ax,
                label=colorbar_label,
                orientation=colorbar_orientation,
            )

    _format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title,
                 xlim=xlim, ylim=ylim, grid=grid)

    return fig, ax, sc


def my_histogram(
    data: ArrayLike,
    ax: Optional[Axes] = None,
    bins: int = 30,
    xlabel: Optional[str] = None,
    ylabel: str = "Frequency",
    title: Optional[str] = None,
    density: bool = False,
    color: Optional[str] = None,
    edgecolor: str = "black",
    alpha: float = 0.85,
    grid: bool = True,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Make a histogram.
    """
    data = _finite_data(data)

    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    hist = ax.hist(
        data,
        bins=bins,
        density=density,
        color=color,
        edgecolor=edgecolor,
        alpha=alpha,
    )

    _format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title,
                 xlim=xlim, ylim=ylim, grid=grid)

    return fig, ax, hist


# ============================================================
# REGULAR GRID MAPS
# ============================================================

def my_contourf(
    x: ArrayLike,
    y: ArrayLike,
    data: ArrayLike,
    shape: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    levels: Union[int, Sequence[float]] = 50,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    colormap: Union[str, object] = DEFAULT_CMAP,
    colorbar_orientation: str = "vertical",
    colorbar_shrink: float = 0.85,
    colorbar_pad: float = 0.04,
    colorbar_aspect: float = 30.0,
    colorbar_ticks: Optional[Sequence[float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = False,
    extend: str = "both",
    grid: bool = True,
    equal_aspect: bool = False,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Filled contour map for regular gridded data.

    Only the x/y axis labels and colorbar label need to be changed in most cases.
    """
    X, Y, Z = _reshape_grid(x, y, data, shape=shape)
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    cmap = my_get_colormap(colormap)
    norm = my_get_norm(Z, vmin=vmin, vmax=vmax, center_zero=center_zero)

    cf = ax.contourf(
        X,
        Y,
        Z,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend=extend,
    )

    my_add_colorbar(
        cf,
        ax=ax,
        label=colorbar_label,
        orientation=colorbar_orientation,
        shrink=colorbar_shrink,
        pad=colorbar_pad,
        aspect=colorbar_aspect,
        ticks=colorbar_ticks,
    )

    _format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title,
                 xlim=xlim, ylim=ylim, grid=grid,
                 equal_aspect=equal_aspect)

    return fig, ax, cf


def my_contour(
    x: ArrayLike,
    y: ArrayLike,
    data: ArrayLike,
    shape: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    levels: Union[int, Sequence[float]] = 15,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    colors: Union[str, Sequence[str]] = "black",
    linewidths: float = 0.6,
    linestyles: str = "-",
    clabel: bool = True,
    clabel_fmt: str = "%.2f",
    clabel_fontsize: int = 8,
    grid: bool = True,
    equal_aspect: bool = False,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Contour-line map for regular gridded data.
    """
    X, Y, Z = _reshape_grid(x, y, data, shape=shape)
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    cs = ax.contour(
        X,
        Y,
        Z,
        levels=levels,
        colors=colors,
        linewidths=linewidths,
        linestyles=linestyles,
    )

    if clabel is True:
        ax.clabel(cs, inline=True, fontsize=clabel_fontsize, fmt=clabel_fmt)

    _format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title,
                 xlim=xlim, ylim=ylim, grid=grid,
                 equal_aspect=equal_aspect)

    return fig, ax, cs


def my_pcolormesh(
    x: ArrayLike,
    y: ArrayLike,
    data: ArrayLike,
    shape: Optional[Tuple[int, int]] = None,
    ax: Optional[Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    colormap: Union[str, object] = DEFAULT_CMAP,
    shading: str = "auto",
    colorbar_orientation: str = "vertical",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = False,
    grid: bool = False,
    equal_aspect: bool = False,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
):
    """
    Pseudocolor map for regular grids.
    """
    X, Y, Z = _reshape_grid(x, y, data, shape=shape)
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    cmap = my_get_colormap(colormap)
    norm = my_get_norm(Z, vmin=vmin, vmax=vmax, center_zero=center_zero)

    pm = ax.pcolormesh(
        X,
        Y,
        Z,
        cmap=cmap,
        norm=norm,
        shading=shading,
    )

    my_add_colorbar(pm, ax=ax, label=colorbar_label,
                    orientation=colorbar_orientation)

    _format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title,
                 grid=grid, equal_aspect=equal_aspect)

    return fig, ax, pm


# ============================================================
# IRREGULAR DATA MAPS
# ============================================================

def my_tricontourf(
    x: ArrayLike,
    y: ArrayLike,
    data: ArrayLike,
    ax: Optional[Axes] = None,
    levels: Union[int, Sequence[float]] = 50,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    colormap: Union[str, object] = DEFAULT_CMAP,
    colorbar_orientation: str = "vertical",
    colorbar_shrink: float = 0.85,
    colorbar_pad: float = 0.04,
    colorbar_aspect: float = 30.0,
    colorbar_ticks: Optional[Sequence[float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = False,
    extend: str = "both",
    grid: bool = True,
    equal_aspect: bool = False,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
):
    """
    Filled triangular contour map for irregular data.
    """
    x, y, data = _check_same_shape(x, y, data)
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    cmap = my_get_colormap(colormap)
    norm = my_get_norm(data, vmin=vmin, vmax=vmax, center_zero=center_zero)

    cf = ax.tricontourf(
        x.ravel(),
        y.ravel(),
        data.ravel(),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend=extend,
    )

    my_add_colorbar(
        cf,
        ax=ax,
        label=colorbar_label,
        orientation=colorbar_orientation,
        shrink=colorbar_shrink,
        pad=colorbar_pad,
        aspect=colorbar_aspect,
        ticks=colorbar_ticks,
    )

    _format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title,
                 grid=grid, equal_aspect=equal_aspect)

    return fig, ax, cf


def my_tricontour(
    x: ArrayLike,
    y: ArrayLike,
    data: ArrayLike,
    ax: Optional[Axes] = None,
    levels: Union[int, Sequence[float]] = 15,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    colors: Union[str, Sequence[str]] = "black",
    linewidths: float = 0.6,
    linestyles: str = "-",
    clabel: bool = True,
    clabel_fmt: str = "%.2f",
    clabel_fontsize: int = 8,
    grid: bool = True,
    equal_aspect: bool = False,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
):
    """
    Triangular contour-line map for irregular data.
    """
    x, y, data = _check_same_shape(x, y, data)
    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)

    cs = ax.tricontour(
        x.ravel(),
        y.ravel(),
        data.ravel(),
        levels=levels,
        colors=colors,
        linewidths=linewidths,
        linestyles=linestyles,
    )

    if clabel is True:
        ax.clabel(cs, inline=True, fontsize=clabel_fontsize, fmt=clabel_fmt)

    _format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title,
                 grid=grid, equal_aspect=equal_aspect)

    return fig, ax, cs


# ============================================================
# CARTOPY MAPS
# ============================================================

def _import_cartopy():
    """
    Import Cartopy only when geographic maps are requested.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except Exception as exc:
        raise ImportError(
            "Cartopy is required for geographic maps. "
            "Install it with: conda install -c conda-forge cartopy"
        ) from exc


def _infer_area(lon: ArrayLike, lat: ArrayLike, margin: float = 0.0):
    """
    Infer geographic area from longitude and latitude.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    return [
        float(np.nanmin(lon) - margin),
        float(np.nanmax(lon) + margin),
        float(np.nanmin(lat) - margin),
        float(np.nanmax(lat) + margin),
    ]


def my_cartopy_basemap(
    area: AreaLike,
    ax=None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    land_color: str = "#d8c28a",
    ocean_color: str = "#9ecae1",
    coastline_color: str = "black",
    border_color: str = "0.25",
    state_color: str = "0.35",
    coastline_linewidth: float = 0.7,
    border_linewidth: float = 0.5,
    state_linewidth: float = 0.4,
    draw_land: bool = True,
    draw_ocean: bool = True,
    draw_coastlines: bool = True,
    draw_countries: bool = True,
    draw_states: bool = True,
    tick_step: float = 1.0,
    show_gridlines: bool = True,
    gridline_color: str = "0.5",
    gridline_alpha: float = 0.45,
    gridline_linewidth: float = 0.4,
    title: Optional[str] = None,
):
    """
    Create a Cartopy basemap from longitude and latitude limits.

    Parameters
    ----------
    area : list
        [lon_min, lon_max, lat_min, lat_max].
    tick_step : float
        Tick interval in degrees. Common values: 1.0 or 2.0.

    Labels
    ------
    Longitude/latitude labels are shown only at the top and left.
    """
    ccrs, cfeature = _import_cartopy()

    lon_min, lon_max, lat_min, lat_max = area
    projection = ccrs.PlateCarree()

    fig, ax = _get_fig_ax(ax=ax, figsize=figsize, projection=projection)

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)

    if draw_ocean is True:
        ax.add_feature(
            cfeature.OCEAN.with_scale("50m"),
            facecolor=ocean_color,
            edgecolor="none",
            zorder=0,
        )

    if draw_land is True:
        ax.add_feature(
            cfeature.LAND.with_scale("50m"),
            facecolor=land_color,
            edgecolor="none",
            zorder=1,
        )

    if draw_coastlines is True:
        ax.coastlines(
            resolution="50m",
            color=coastline_color,
            linewidth=coastline_linewidth,
            zorder=5,
        )

    if draw_countries is True:
        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            edgecolor=border_color,
            facecolor="none",
            linewidth=border_linewidth,
            zorder=5,
        )

    if draw_states is True:
        states = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="50m",
            facecolor="none",
        )
        ax.add_feature(
            states,
            edgecolor=state_color,
            linewidth=state_linewidth,
            zorder=5,
        )

    if show_gridlines is True:
        gl = ax.gridlines(
            crs=projection,
            draw_labels=True,
            linewidth=gridline_linewidth,
            color=gridline_color,
            alpha=gridline_alpha,
            linestyle="--",
        )

        gl.top_labels = True
        gl.left_labels = True
        gl.bottom_labels = False
        gl.right_labels = False

        gl.xlocator = MultipleLocator(tick_step)
        gl.ylocator = MultipleLocator(tick_step)

        gl.xlabel_style = {"size": 9}
        gl.ylabel_style = {"size": 9}

    if title is not None:
        ax.set_title(title, fontsize=13)

    return fig, ax


def my_cartopy_scatter(
    lon: ArrayLike,
    lat: ArrayLike,
    values: Optional[ArrayLike] = None,
    area: Optional[AreaLike] = None,
    ax=None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    colormap: Union[str, object] = DEFAULT_CMAP,
    color: str = "red",
    size: float = 20.0,
    marker: str = "o",
    alpha: float = 1.0,
    colorbar: bool = False,
    colorbar_label: Optional[str] = None,
    tick_step: float = 1.0,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = False,
):
    """
    Plot scattered geographic data on a Cartopy basemap.
    """
    lon, lat = _check_same_shape(lon, lat)

    if area is None:
        area = _infer_area(lon, lat, margin=0.0)

    fig, ax = my_cartopy_basemap(
        area=area,
        ax=ax,
        figsize=figsize,
        tick_step=tick_step,
        title=title,
    )

    ccrs, _ = _import_cartopy()
    projection = ccrs.PlateCarree()

    if values is None:
        sc = ax.scatter(
            lon,
            lat,
            s=size,
            color=color,
            marker=marker,
            alpha=alpha,
            transform=projection,
            zorder=10,
        )
    else:
        values = np.asarray(values)
        if values.shape != lon.shape:
            raise ValueError("values must have the same shape as lon and lat.")

        cmap = my_get_colormap(colormap)
        norm = my_get_norm(values, vmin=vmin, vmax=vmax, center_zero=center_zero)

        sc = ax.scatter(
            lon,
            lat,
            c=values,
            s=size,
            cmap=cmap,
            norm=norm,
            marker=marker,
            alpha=alpha,
            transform=projection,
            zorder=10,
        )

        if colorbar is True:
            my_add_colorbar(sc, ax=ax, label=colorbar_label)

    return fig, ax, sc


def my_cartopy_tricontourf(
    lon: ArrayLike,
    lat: ArrayLike,
    data: ArrayLike,
    area: Optional[AreaLike] = None,
    ax=None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    levels: Union[int, Sequence[float]] = 50,
    colormap: Union[str, object] = DEFAULT_CMAP,
    colorbar_label: Optional[str] = None,
    tick_step: float = 1.0,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = False,
    extend: str = "both",
):
    """
    Plot tricontourf geographic data on a Cartopy basemap.
    """
    lon, lat, data = _check_same_shape(lon, lat, data)

    if area is None:
        area = _infer_area(lon, lat, margin=0.0)

    fig, ax = my_cartopy_basemap(
        area=area,
        ax=ax,
        figsize=figsize,
        tick_step=tick_step,
        title=title,
    )

    ccrs, _ = _import_cartopy()
    projection = ccrs.PlateCarree()

    cmap = my_get_colormap(colormap)
    norm = my_get_norm(data, vmin=vmin, vmax=vmax, center_zero=center_zero)

    cf = ax.tricontourf(
        lon.ravel(),
        lat.ravel(),
        data.ravel(),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend=extend,
        transform=projection,
        zorder=3,
    )

    my_add_colorbar(cf, ax=ax, label=colorbar_label)

    return fig, ax, cf


def my_cartopy_tricontour(
    lon: ArrayLike,
    lat: ArrayLike,
    data: ArrayLike,
    area: Optional[AreaLike] = None,
    ax=None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    levels: Union[int, Sequence[float]] = 15,
    colors: str = "black",
    linewidths: float = 0.5,
    linestyles: str = "-",
    clabel: bool = False,
    clabel_fmt: str = "%.2f",
    tick_step: float = 1.0,
    title: Optional[str] = None,
):
    """
    Plot tricontour line geographic data on a Cartopy basemap.
    """
    lon, lat, data = _check_same_shape(lon, lat, data)

    if area is None:
        area = _infer_area(lon, lat, margin=0.0)

    fig, ax = my_cartopy_basemap(
        area=area,
        ax=ax,
        figsize=figsize,
        tick_step=tick_step,
        title=title,
    )

    ccrs, _ = _import_cartopy()
    projection = ccrs.PlateCarree()

    cs = ax.tricontour(
        lon.ravel(),
        lat.ravel(),
        data.ravel(),
        levels=levels,
        colors=colors,
        linewidths=linewidths,
        linestyles=linestyles,
        transform=projection,
        zorder=4,
    )

    if clabel is True:
        ax.clabel(cs, inline=True, fontsize=8, fmt=clabel_fmt)

    return fig, ax, cs


# ============================================================
# SAVE FIGURE
# ============================================================

def my_save_figure(
    filename: str,
    fig: Optional[Figure] = None,
    dpi: int = DEFAULT_DPI,
    bbox_inches: str = "tight",
    transparent: bool = True,
    facecolor: str = "none",
    close: bool = False,
):
    """
    Save a Matplotlib figure.

    Default:
        dpi=300
        bbox_inches="tight"
        transparent=True
    """
    if fig is None:
        fig = plt.gcf()

    folder = os.path.dirname(filename)

    if folder not in ["", "."] and not os.path.exists(folder):
        os.makedirs(folder)

    fig.savefig(
        filename,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
        facecolor=facecolor,
    )

    if close is True:
        plt.close(fig)

    return filename


# ============================================================
# SHORT ALIASES
# ============================================================

my_rectangle = my_draw_rectangle
my_circle = my_draw_circle
my_polygon = my_draw_polygon
my_lineplot = my_plot
my_savefig = my_save_figure
my_basemap = my_cartopy_basemap
my_contour_map = my_contourf
my_tricontour_map = my_tricontourf
