import plotly.graph_objects as go
import geopandas as gpd

def plot_polygon_shapes_interactive(shapefile_path):
    """
    Plots raw 2D polygon shapes from a shapefile using Plotly (no map projection).
    Perfect for fantasy or image-based maps.
    """
    gdf = gpd.read_file(shapefile_path)

    fig = go.Figure()

    for i, row in gdf.iterrows():
        poly = row.geometry
        if poly.is_empty:
            continue
        if poly.geom_type == "Polygon":
            x, y = map(list, poly.exterior.xy)
            
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                fill="toself",
                name=f"ID {row.get('id', i)}",
                hovertext=f"ID: {row.get('id', i)}<br>Type: {row.get('type', 'unknown')}",
                hoverlabel=dict(namelength=0, bgcolor="white", font_size=12),
                hoverinfo="text",
                line=dict(color="#556b2f", width=1),
                fillcolor="rgba(160, 184, 120, 0.5)"
            ))
            
        elif poly.geom_type == "MultiPolygon":
            for j, subpoly in enumerate(poly.geoms):
                x, y = map(list, subpoly.exterior.xy)
            
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines",
                    fill="toself",
                    name=f"ID {row.get('id', i)}.{j}",
                    hovertext=f"ID: {row.get('id', i)}.{j}<br>Type: {row.get('type', 'unknown')}",
                    hoverinfo="text",
                    line=dict(color="#556b2f", width=1),
                    fillcolor="rgba(160, 184, 120, 0.5)"
                ))

    fig.update_layout(
        title="Htrea - Polygon Viewer",
        xaxis_title="X",
        yaxis_title="Y",
        
        #more "data-ie" view
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(autorange="reversed"),  # match image space
        
        #more game-y view
        # yaxis=dict(scaleanchor=None),
        # xaxis=dict(scaleanchor=None),
        
        plot_bgcolor="#f4f1e1",
        paper_bgcolor="#f4f1e1",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    fig.update_yaxes(autorange='reversed')

    return fig

def apply_style(fig, style="fantasy"):
    if style == "fantasy":
        fig.update_layout(...)  # earthy tone, fantasy fonts
    elif style == "data":
        fig.update_layout(...)  # minimal grid, white bg, etc.
    return fig