import numpy as np
import open3d as o3d

PLY_FILE = "test4.ply"


# Helpers
def pts(g):
    return np.asarray(
        g.vertices if isinstance(g, o3d.geometry.TriangleMesh) else g.points
    )


def sphere(p, c, r=2.0):
    s = o3d.geometry.TriangleMesh.create_sphere(r)
    s.paint_uniform_color(c)
    s.translate(p)
    s.compute_vertex_normals()
    return s


def align_to_y(g):
    p = pts(g)
    g.translate(-p.mean(axis=0))
    _, v = np.linalg.eigh(np.cov(p.T))
    axis = v[:, -1]
    if axis[1] < 0:
        axis *= -1
    cross = np.cross(axis, [0, 1, 0])
    s = np.linalg.norm(cross)
    if s < 1e-8:
        return g
    vx = np.array(
        [[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]]
    )
    R = np.eye(3) + vx + vx @ vx * ((1 - axis[1]) / s**2)
    g.rotate(R, center=(0, 0, 0))
    return g


# Load geometry
geom = o3d.io.read_triangle_mesh(PLY_FILE)
if geom.has_triangles():
    geom.compute_vertex_normals()
    geom.remove_duplicated_vertices()
    geom.remove_degenerate_triangles()
    geom.paint_uniform_color([0.92, 0.92, 0.88])
else:
    geom = o3d.io.read_point_cloud(PLY_FILE)
    geom.estimate_normals()

# Orientation
geom = align_to_y(geom)
p = pts(geom)

# Lowest point
low = p[np.argmin(p[:, 1])]

# Rotate so lowest faces +Z
angle = np.arctan2(low[0], low[2])
geom.rotate(geom.get_rotation_matrix_from_axis_angle([0, -angle, 0]), center=(0, 0, 0))
p = pts(geom)

# Landmarks
ymin, ymax = p[:, 1].min(), p[:, 1].max()
zmin, zmax = p[:, 2].min(), p[:, 2].max()
x_min, x_max = p[:, 0].min(), p[:, 0].max()
h = ymax - ymin

# Bottom 15% → most left
bottom = p[p[:, 1] <= ymin + 0.15 * h]
left = bottom[np.argmin(bottom[:, 0])]

# Top along Y-axis
axis_pts = p[np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) <= 1.0]
top_axis = axis_pts[np.argmax(axis_pts[:, 1])] if len(axis_pts) else None

# Radial tuberosity
roi = p[(p[:, 1] >= ymax - 0.20 * h) & (p[:, 1] <= ymax - 0.08 * h)]
shaft_xz = roi[:, [0, 2]].mean(axis=0)
shaft = np.array([shaft_xz[0], roi[:, 1].mean(), shaft_xz[1]])
angles = np.degrees(np.arctan2(roi[:, 2] - shaft[2], roi[:, 0] - shaft[0]))
best = max(
    ((angles >= a) & (angles <= a + 120) for a in np.arange(-180, 180, 10)),
    key=lambda m: m.sum(),
)
tub = roi[best]
tub_pt = tub[np.argmax(np.linalg.norm(tub[:, [0, 2]] - shaft[[0, 2]], axis=1))]

# Deep + low point (yellow)
y_norm = (p[:, 1] - ymin) / h
z_norm = (p[:, 2] - zmin) / (zmax - zmin)
score = 0.8 * y_norm + 0.2 * z_norm
deep_low = p[np.argmin(score)]

# Deep + low + right point (cyan)
x_norm = (p[:, 0] - x_min) / (x_max - x_min)
score_xyz = 0.6 * y_norm + 0.25 * z_norm - 0.15 * x_norm
deep_low_right = p[np.argmin(score_xyz)]

# Visualization
vis = o3d.visualization.Visualizer()
vis.create_window("3D Bone Viewer", 1200, 900)

vis.add_geometry(geom)
vis.add_geometry(sphere(low, [1, 0, 0]))  # rood
vis.add_geometry(sphere(left, [0, 0, 1]))  # blauw
vis.add_geometry(sphere(tub_pt, [1, 0, 1], 2.5))  # paars
vis.add_geometry(sphere(deep_low, [1, 1, 0]))  # geel
vis.add_geometry(sphere(deep_low_right, [0, 1, 1]))  # cyaan
if top_axis is not None:
    vis.add_geometry(sphere(top_axis, [0, 1, 0]))  # groen

opt = vis.get_render_option()
opt.background_color = [0.05, 0.05, 0.05]
opt.light_on = True
opt.mesh_show_back_face = True
opt.point_size = 2
vis.get_view_control().set_zoom(0.8)

vis.run()
vis.destroy_window()
