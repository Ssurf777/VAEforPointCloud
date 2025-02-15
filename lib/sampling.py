import numpy as np
import random

class PointSampler(object):
    """Class to sample points from mesh triangles."""
    def __init__(self, output_size):
        assert isinstance(output_size, int), "Output size must be an integer."
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        """Calculates the area of a triangle given its vertices."""
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        """Samples a point uniformly from the surface of a triangle."""
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        """Samples points from a mesh defined by vertices and faces."""
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros(len(faces))

        for i in range(len(areas)):
            areas[i] = self.triangle_area(
                verts[faces[i][0]], verts[faces[i][1]], verts[faces[i][2]]
            )

        sampled_faces = random.choices(faces, weights=areas, k=self.output_size)
        sampled_points = np.zeros((self.output_size, 3))
        
        for i in range(len(sampled_faces)):
            sampled_points[i] = self.sample_point(
                verts[sampled_faces[i][0]],
                verts[sampled_faces[i][1]],
                verts[sampled_faces[i][2]]
            )
        
        return sampled_points
