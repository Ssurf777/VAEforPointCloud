import numpy as np

def read_off(file):
    """Reads an OFF file and returns vertices and faces."""
    if 'OFF' != file.readline().strip():
        raise ValueError('Not a valid OFF header')
    n_verts, n_faces, _ = map(int, file.readline().strip().split())
    verts = [list(map(float, file.readline().strip().split())) for _ in range(n_verts)]
    faces = [list(map(int, file.readline().strip().split()))[1:] for _ in range(n_faces)]
    return verts, faces
