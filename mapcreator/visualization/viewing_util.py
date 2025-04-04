from pathlib import Path

def save_figure_to_html(fig, output_html):
    """
    Saves a Plotly figure to an interactive HTML file.

    Args:
        fig (plotly.graph_objs.Figure): The figure to save.
        output_html (Path or str): Output path.
    """
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))
    print(f"✅ Saved HTML to: {output_html}")