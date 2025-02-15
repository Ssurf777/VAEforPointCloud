import plotly.graph_objs as go
from lib.visualization import visualize_rotate

def pcshow(xs, ys, zs):
    """Displays a 3D point cloud with rotation animation."""
    data = [go.Scatter3d(x=xs, y=ys, z=zs, mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(
        marker=dict(size=2, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )
    fig.show()
