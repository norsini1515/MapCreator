import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.widgets import Button

def plot_single_polygon(polygon, index=0):
    """
    Plot a single polygon to check if visualization works.
    
    Args:
        polygon: A Shapely Polygon.
        index: Polygon index for labeling.
    """
    if polygon.is_empty:
        print(f"⚠️ Polygon {index} is EMPTY, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    x, y = polygon.exterior.xy
    ax.fill(x, y, edgecolor="black", facecolor="lightblue", linewidth=1)
    
    ax.set_title(f"Polygon {index}")
    plt.show()

def plot_polygons_separately(shapefile_path):
    """
    Plots each polygon individually in separate figures.

    Args:
        shapefile_path: Path to the shapefile.
    
    Returns:
        List of polygon indices that failed to plot.
    """
    problem_polygons = []
    
    gdf = gpd.read_file(shapefile_path)

    if gdf.empty:
        raise ValueError("Shapefile is empty or invalid.")

    print(f"Total Polygons: {len(gdf)}")

    for i, row in gdf.iterrows():
        print(f"Plotting Polygon {i}...")
        try:
            plot_single_polygon(row.geometry, i)  # Pass Shapely Polygon directly
        except Exception as e:
            print(f"Error plotting polygon {i}: {e}")
            problem_polygons.append(i)

    return problem_polygons

def plot_all_polygons(polygons, title=None):
    """
    Plots all polygons in a single Matplotlib figure.
    
    Args:
        polygons: List of Shapely Polygons.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, polygon in enumerate(polygons):
        x, y = polygon.exterior.xy
        ax.fill(x, y, edgecolor="black", facecolor="lightblue", linewidth=1)
    
    if not title:
        plt.title("All Polygons - Single Plot")
    else: 
        plt.title(title)
        
    plt.show()

class PolygonViewer:
    """
    Interactive Polygon Viewer for inspecting shapefile polygons one at a time.
    Allows navigation with both keyboard arrows and clickable buttons.
    """
    def __init__(self, shapefile_path, detect_problems=True):
        self.gdf = gpd.read_file(shapefile_path)
        self.index = 0  # Start at the first polygon
        self.problematic_polygons = []  # Store indices of polygons that fail
        
        if self.gdf.empty:
            raise ValueError("Shapefile is empty or invalid.")

        # Run problem detection before launching viewer
        if detect_problems:
            self.detect_problematic_polygons()

        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.2)  # Add space for buttons
        
        # Create buttons
        self.ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])  # Position for "Previous"
        self.ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])  # Position for "Next"

        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_next = Button(self.ax_next, 'Next')

        self.btn_prev.on_clicked(self.prev_polygon)
        self.btn_next.on_clicked(self.next_polygon)

        # Enable keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.plot_polygon()  # Plot the first polygon
        plt.show()

    def detect_problematic_polygons(self):
        """Detects polygons that might be problematic without plotting them."""
        print("\n🔎 Running polygon integrity check...")

        for i, row in self.gdf.iterrows():
            polygon = row.geometry
            try:
                if polygon.is_empty:
                    raise ValueError("Polygon is empty")
                if not polygon.is_valid:
                    raise ValueError("Polygon is invalid")
                if len(polygon.exterior.coords) < 3:
                    raise ValueError("Polygon has too few points")
            except Exception as e:
                print(f"⚠️ Problem with Polygon {i}: {e}")
                self.problematic_polygons.append(i)

        if self.problematic_polygons:
            print(f"⚠️ Found {len(self.problematic_polygons)} problematic polygons: {self.problematic_polygons}")
        else:
            print("✅ No problematic polygons detected.")

    def plot_polygon(self):
        """Plots the current polygon."""
        self.ax.clear()
        polygon = self.gdf.iloc[self.index].geometry
        
        try:
            if polygon.is_empty:
                raise ValueError("Polygon is empty")

            x, y = polygon.exterior.xy
            self.ax.fill(x, y, edgecolor="black", facecolor="lightblue", linewidth=1)
            self.ax.set_title(f"Polygon {self.index+1} of {len(self.gdf)}")
        except Exception as e:
            print(f"⚠️ Error plotting polygon {self.index}: {e}")
            self.problematic_polygons.append(self.index)
            self.ax.set_title(f"Polygon {self.index+1} - ERROR")

        self.fig.canvas.draw()

    def on_key_press(self, event):
        """Handles key press events for keyboard navigation."""
        if event.key == 'right':
            self.next_polygon(None)
        elif event.key == 'left':
            self.prev_polygon(None)

    def next_polygon(self, event):
        """Go to the next polygon."""
        self.index = (self.index + 1) % len(self.gdf)
        self.plot_polygon()

    def prev_polygon(self, event):
        """Go to the previous polygon."""
        self.index = (self.index - 1) % len(self.gdf)
        self.plot_polygon()

def debug_shapefile_interactive(shapefile_path):
    """
    Launches an interactive viewer for cycling through polygons in a shapefile.

    Args:
        shapefile_path: Path to the shapefile.

    Returns:
        List of indices of problematic polygons.
    """
    viewer = PolygonViewer(shapefile_path, detect_problems=True)
    return viewer.problematic_polygons
