import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import pandas as pd
import numpy as np

def plot_rays(rays_gdf):
    fig, ax = plt.subplots(figsize=(10, 8))
    rays_gdf.plot(ax=ax, linewidth=0.3, color='orange', alpha=0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Flip it upright for image-style view
    plt.title("Ray Paths")
    plt.tight_layout()
    plt.show()
    
def plot_histogram_by_group(
    df: pd.DataFrame,
    column: str,
    group: str,
    bins: int = 50,
    title: str = None,
    xlabel: str = None,
    ylabel: str = "Density",
    alpha: float = 0.7,
    kde: bool = False,
    stat: str = "density",
    figsize: tuple = (10, 5),
    show: bool = True
):
    plt.figure(figsize=figsize)
    for g in df[group].unique():
        subset = df[df[group] == g]
        sns.histplot(subset[column], bins=bins, label=str(g), kde=kde, stat=stat, alpha=alpha)
    plt.title(title or f"{column} Distribution by {group}")
    plt.xlabel(xlabel or column)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()

def plot_polar_histogram_by_group(
    df: pd.DataFrame,
    angle_col: str,
    group_col: str,
    bins: int = 72,
    title: str = "Polar Histogram",
    figsize: tuple = (6, 6),
    show: bool = True
):
    plt.figure(figsize=figsize)
    ax = plt.subplot(111, polar=True)
    for group in df[group_col].unique():
        angles = df[df[group_col] == group][angle_col]
        ax.hist(angles, bins=bins, alpha=0.6, label=str(group))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    if show:
        plt.show()

def plot_scatter_by_group(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    alpha: float = 0.5,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: tuple = (10, 5),
    show: bool = True
):
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x, y=y, hue=group, alpha=alpha)
    plt.title(title or f"{y} vs {x}")
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()
