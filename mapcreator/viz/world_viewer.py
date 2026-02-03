import plotly.graph_objects as go
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, LineString, MultiPoint, MultiLineString

from mapcreator import configs, directories
from mapcreator.viz import viewing_util

TYPE_COLOR_MAP = {
    "land": "rgba(160, 184, 120, 0.5)",             # muted green
    "lakes": "rgba(100, 160, 220, 0.5)",            # soft blue
    "ocean": "rgba(34, 102, 102, 0.58)",            # sea-foam tone
    "connected_feature": "rgba(70, 130, 180, 0.5)", # steel blue
    "unknown": "rgba(200, 200, 200, 0.6)"           # gray fallback
}
TYPE_OUTLINE_COLOR_MAP = {
    "land": "rgba(160, 184, 120, 0.7)",             # muted green
    "lakes": "rgba(100, 160, 220, 0.7)",            # soft blue
    "ocean": "rgba(34, 102, 102, 0.78)",            # sea-foam tone
    "connected_feature": "rgba(70, 130, 180, 0.7)", # steel blue
    "unknown": "rgba(200, 200, 200, 0.8)"           # gray fallback
}


def plot_shapes(data, fig=None):
    """
    Plots raw 2D polygon shapes (including holes) using Plotly.

    Args:
        data (Path | GeoDataFrame): Path to a shapefile or an in-memory GeoDataFrame.
        fig (go.Figure, optional): Existing Plotly figure to plot onto.

    Returns:
        go.Figure: The Plotly figure with shapes plotted.
    """
    if isinstance(data, (str, Path)):
        gdf = gpd.read_file(data)
    elif isinstance(data, gpd.GeoDataFrame):
        gdf = data
    else:
        raise TypeError("Input must be a Path or a GeoDataFrame.")

    if not fig:
        fig = go.Figure()

    if "id" not in gdf.columns:
        gdf["id"] = gdf.index

    for i, row in gdf.iterrows():
        poly = row.geometry
        if poly.is_empty:
            continue

        hover_text = f"ID: {row.get('id', i)}<br>Type: {row.get('type', 'unknown')}"
        fill_color = TYPE_COLOR_MAP.get(row.get("type", "land"))
        outline_color = TYPE_OUTLINE_COLOR_MAP.get(row.get("type", "land"))
        
        def plot_ring(x, y, filled=True, fill_color=fill_color):
            fig.add_trace(go.Scatter(
                x=list(x), y=list(y),
                mode="lines",
                fill="toself" if filled else None,
                name=f"ID {row['id']}",
                hovertext=hover_text,
                hoverlabel=dict(namelength=0, bgcolor="white", font_size=12),
                hoverinfo="text",
                line=dict(color=outline_color, width=1),
                fillcolor=fill_color if filled else None
            ))

        if poly.geom_type == "Polygon":
            x, y = poly.exterior.xy
            plot_ring(x, y, filled=True, fill_color=fill_color) 
            
            for ring in poly.interiors:
                x, y = ring.xy
                plot_ring(x, y, filled=True, fill_color="#f4f1e1")
            
        elif poly.geom_type == "MultiPolygon":
            for j, subpoly in enumerate(poly.geoms):
                x, y = subpoly.exterior.xy
                plot_ring(x, y, filled=True)
                
                for ring in subpoly.interiors:
                    x, y = ring.xy
                    plot_ring(x, y, filled=False)

    fig.update_layout(
        title=f"{configs.WORLD_NAME} - Polygon Viewer",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(autorange="reversed"),  # Ensure proper image-space orientation
        plot_bgcolor="#f4f1e1",
        paper_bgcolor="#f4f1e1",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )

    fig.update_yaxes(autorange='reversed')  # Reinforce reversal in case layout doesn’t apply it

    return fig

def plot_overlay(fig, overlay_data, color="red", name="overlay", size=6, width=2):
    """
    Adds overlay markers or lines to an existing Plotly figure.

    Args:
        fig (go.Figure): The base figure to plot onto.
        overlay_data (GeoDataFrame | list[Point|LineString]): The overlay geometries.
        color (str): Color of the overlay (default: red).
        name (str): Name for the trace (optional).
        size (int): Marker size for points.
        width (int): Line width for line features.

    Returns:
        go.Figure: The figure with overlays added.
    """
    if isinstance(overlay_data, gpd.GeoDataFrame):
        geoms = overlay_data.geometry
    else:
        geoms = overlay_data

    for i, geom in enumerate(geoms):
        if isinstance(geom, (Point, MultiPoint)):
            x, y = zip(*[(pt.x, pt.y) for pt in geom.geoms] if isinstance(geom, MultiPoint) else [(geom.x, geom.y)])
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="markers",
                marker=dict(color=color, size=size),
                name=f"{name}_{i}",
                showlegend=False
            ))

        elif isinstance(geom, (LineString, MultiLineString)):
            lines = geom.geoms if isinstance(geom, MultiLineString) else [geom]
            for j, line in enumerate(lines):
                x, y = map(list, line.xy)
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode="lines",
                    line=dict(color=color, width=width),
                    name=f"{name}_{i}_{j}",
                    showlegend=False
                ))

    return fig

def apply_style(fig, style="fantasy"):
    if style == "fantasy":
        fig.update_layout(
            xaxis=dict(scaleanchor=None),
            yaxis=dict(scaleanchor=None),
        )

    elif style == "data":
        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(autorange="reversed"),  # match image space
        )
    return fig

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    geojson_path = Path("C:/Users/nicho/Documents/World Building/MapCreator/data/processed/vectors")
    file = "base_20260202_2212.geojson"
    geojson_path = geojson_path / file
    gdf = gpd.read_file(geojson_path)
    print(gdf.columns.tolist())
    print(gdf.head(5))

    
    ax = gdf.plot(
        color=gdf["color"],
        edgecolor="black",
        linewidth=0.2
    )
    plt.show()


    # exit()
    # fig = plot_shapes(gdf)

    # html_path = directories.TEMP_FILES_DIR / f"{shapefile_name}.html"
    # viewing_util.save_figure_to_html(fig, html_path, open_on_export=True)