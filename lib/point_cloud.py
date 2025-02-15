import plotly.graph_objs as go

def visualize_rotate(data):
    """Creates a rotating 3D plot using Plotly."""
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames = []

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))

    fig = go.Figure(data=data, layout=go.Layout(frames=frames))
    return fig
    
def pcshow(xs, ys, zs):
    """Displays a 3D point cloud with rotation animation."""
    data = [go.Scatter3d(x=xs, y=ys, z=zs, mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(
        marker=dict(size=2, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )
    fig.show()
