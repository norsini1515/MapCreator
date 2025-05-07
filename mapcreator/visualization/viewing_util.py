from pathlib import Path
import subprocess
import sys

def save_figure_to_html(fig, output_html, open_on_export=False):
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
    
    if open_on_export:
        # Open the HTML file in the default browser
        try:
            if sys.platform == "win32":
                subprocess.Popen(["start", str(output_html)], shell=True)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(output_html)])
            else:
                subprocess.Popen(["xdg-open", str(output_html)])
        except Exception as e:
            print(f"Could not open HTML file automatically: {e}")