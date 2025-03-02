"""
STL Text File Reader

This script reads an ASCII STL file and extracts the vertex coordinates and face indices.

Usage:
    with open('filename.stl', 'r') as file:
        vertices, faces = read_stl_text(file)

Returns:
    - vertices (list of tuples): A list of unique vertex coordinates (x, y, z) as float tuples.
    - faces (list of lists): A list of faces, each represented as a list of three vertex indices.
"""

def read_stl_text(file):
    """
    Reads an ASCII STL file and extracts vertex coordinates and face indices.

    Args:
        file (file object): A file object opened in text mode containing an ASCII STL file.

    Returns:
        tuple: A tuple containing:
            - vertices (list of tuples): A list of unique vertex coordinates (x, y, z) as float tuples.
            - faces (list of lists): A list of faces, each represented as a list of three vertex indices.
    """
    verts = []  # List to store unique vertex coordinates
    faces = []  # List to store faces (triangles as indices)
    vert_map = {}  # Dictionary to map vertex coordinates to their index
    current_face = []  # Temporary list to store the current face's vertices

    for line in file:
        parts = line.strip().split()
        if not parts:
            continue
        
        if parts[0] == 'vertex':
            vertex = tuple(float(p) for p in parts[1:])
            if vertex not in vert_map:
                vert_map[vertex] = len(verts)
                verts.append(vertex)
            current_face.append(vert_map[vertex])
            
            if len(current_face) == 3:  # A face consists of exactly 3 vertices
                faces.append(current_face)
                current_face = []  # Reset for the next face

    return verts, faces
"""
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python stl_reader.py <stl_file>")
        sys.exit(1)
    
    try:
        with open(sys.argv[1], 'r') as file:
            vertices, faces = read_stl_text(file)
        print(f"Read {len(vertices)} vertices and {len(faces)} faces.")
    except Exception as e:
        print(f"Error reading file: {e}")
"""
