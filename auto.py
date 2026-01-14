import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

PLY_FILE = "test2raw.ply"  # Bestand met 3D botmodel


# -----------------------------
# Functies
# -----------------------------
def pts(g):
    return np.asarray(
        g.vertices if isinstance(g, o3d.geometry.TriangleMesh) else g.points
    )


def sphere(p, c, r=2.5):
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


def fit_sphere(points):
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    A = np.c_[2 * X, 2 * Y, 2 * Z, np.ones(len(X))]
    f = X**2 + Y**2 + Z**2
    C, residuals, _, _ = np.linalg.lstsq(A, f, rcond=None)
    center = C[:3]
    radius = np.sqrt(C[3] + np.sum(center**2))
    rmse = np.sqrt(np.mean((np.linalg.norm(points - center, axis=1) - radius) ** 2))
    return center, radius, rmse


# -----------------------------
# 1. Mesh inlezen
# -----------------------------
geom = o3d.io.read_triangle_mesh(PLY_FILE)
if geom.has_triangles():
    geom.compute_vertex_normals()
    geom.remove_duplicated_vertices()
    geom.remove_degenerate_triangles()
    geom.paint_uniform_color([0.92, 0.92, 0.88])
else:
    geom = o3d.io.read_point_cloud(PLY_FILE)
    geom.estimate_normals()

geom = align_to_y(geom)
p = pts(geom)

# --- Op-zijn-kop fix ---
ymin, ymax = p[:, 1].min(), p[:, 1].max()
h = ymax - ymin
bottom = p[p[:, 1] <= ymin + 0.20 * h]
top = p[p[:, 1] >= ymax - 0.20 * h]
if np.std(bottom[:, [0, 2]]) < np.std(top[:, [0, 2]]):
    geom.rotate(
        geom.get_rotation_matrix_from_axis_angle([np.pi, 0, 0]), center=(0, 0, 0)
    )
    p = pts(geom)

# --- Laagste punt en rotatie naar +Z ---
low = p[np.argmin(p[:, 1])]
angle = np.arctan2(low[0], low[2])
geom.rotate(geom.get_rotation_matrix_from_axis_angle([0, -angle, 0]), center=(0, 0, 0))
p = pts(geom)

ymin, ymax = p[:, 1].min(), p[:, 1].max()
zmin, zmax = p[:, 2].min(), p[:, 2].max()
x_min, x_max = p[:, 0].min(), p[:, 0].max()
h = ymax - ymin

# -----------------------------
# 2. Detecteer linker/rechterarm
# -----------------------------
y_norm = (p[:, 1] - ymin) / h
z_norm = (p[:, 2] - zmin) / (zmax - zmin)
score_full = 0.8 * y_norm + 0.2 * z_norm
deep_low = p[np.argmin(score_full)]
left_arm = deep_low[0] < 0
print("Linker arm" if left_arm else "Rechter arm")

# -----------------------------
# 3. Tijdelijke spiegel voor consistente berekeningen
# -----------------------------
p_mirror = p.copy()
if left_arm:
    p_mirror[:, 0] *= -1

# -----------------------------
# 4. Landmarks
# -----------------------------
# Blauw (Lister's tubercle) met ratio -X voorkeur
bottom_roi = p_mirror[p_mirror[:, 1] <= ymin + 0.15 * h]
score_blue = (bottom_roi[:, 0] - x_min) / (x_max - x_min) * 0.8 + (
    bottom_roi[:, 2] - zmin
) / (zmax - zmin) * 0.2
left = bottom_roi[np.argmin(score_blue)]

# Paars (Radial tuberosity)
roi = p_mirror[
    (p_mirror[:, 1] >= ymax - 0.20 * h) & (p_mirror[:, 1] <= ymax - 0.08 * h)
]
shaft_xz = roi[:, [0, 2]].mean(axis=0)
shaft = np.array([shaft_xz[0], roi[:, 1].mean(), shaft_xz[1]])
angles = np.degrees(np.arctan2(roi[:, 2] - shaft[2], roi[:, 0] - shaft[0]))
best = max(
    ((angles >= a) & (angles <= a + 120) for a in np.arange(-180, 180, 10)),
    key=lambda m: m.sum(),
)
tub = roi[best]
tub_pt = tub[np.argmax(np.linalg.norm(tub[:, [0, 2]] - shaft[[0, 2]], axis=1))]

# Geel (Peak dorsal rim)
score_yz = 0.8 * y_norm + 0.2 * z_norm
deep_low = p[np.argmin(score_yz)]

# Cyaan (Peak volar rim)
x_norm_full = (p_mirror[:, 0] - x_min) / (x_max - x_min)
score_xyz = 0.6 * y_norm + 0.15 * z_norm - 0.25 * x_norm_full
deep_low_right = p_mirror[np.argmin(score_xyz)]


# Zet landmarks terug naar originele coördinaten
if left_arm:
    left[0] *= -1
    tub_pt[0] *= -1
    deep_low[0] *= -1
    deep_low_right[0] *= -1

# -----------------------------
# 5. Kuilen onderaan via bolfit
# -----------------------------
vertices = np.asarray(geom.vertices)
normals = np.asarray(geom.vertex_normals)

roi_fraction = 0.07
neighborhood_radius = 6
n_kuilen = 2
min_dist = 8

y_limit = ymin + roi_fraction * h
roi_mask = vertices[:, 1] <= y_limit
roi_points = vertices[roi_mask][::5]
roi_normals = normals[roi_mask][::5]

xz_center = roi_points[:, [0, 2]].mean(axis=0)
bottom_mask = np.linalg.norm(
    roi_points[:, [0, 2]] - xz_center, axis=1
) <= 0.55 * np.max(np.linalg.norm(roi_points[:, [0, 2]] - xz_center, axis=1))
roi_points = roi_points[bottom_mask]
roi_normals = roi_normals[bottom_mask]

tree = cKDTree(roi_points[:, [0, 2]])
sphere_scores = []
for i, pt in enumerate(roi_points):
    idx = tree.query_ball_point([pt[0], pt[2]], neighborhood_radius)
    if len(idx) < 5:
        continue
    local_points = roi_points[idx]
    center, radius, rmse = fit_sphere(local_points)
    sphere_scores.append((pt, rmse))
sphere_scores.sort(key=lambda x: x[1])

selected_pts = []
for pt, score in sphere_scores:
    if all(np.linalg.norm(pt[[0, 2]] - p[[0, 2]]) >= min_dist for p in selected_pts):
        _, idx_mesh = cKDTree(p[:, [0, 2]]).query([pt[0], pt[2]])
        surface_pt = np.array([pt[0], p[idx_mesh, 1], pt[2]])
        selected_pts.append(surface_pt)
    if len(selected_pts) >= n_kuilen:
        break

# -----------------------------
# 6. Bolfit bovenste 5%
# -----------------------------
roi_fraction_top = 0.05
y_limit_top = ymax - roi_fraction_top * h
top_mask = vertices[:, 1] >= y_limit_top
top_points = vertices[top_mask][::5]

tree_top = cKDTree(top_points[:, [0, 2]])
sphere_scores_top = []
for i, pt in enumerate(top_points):
    idx = tree_top.query_ball_point([pt[0], pt[2]], neighborhood_radius)
    if len(idx) < 5:
        continue
    local_points = top_points[idx]
    center, radius, rmse = fit_sphere(local_points)
    sphere_scores_top.append((pt, rmse))
sphere_scores_top.sort(key=lambda x: x[1])

top_bol = sphere_scores_top[0][0] if len(sphere_scores_top) > 0 else None

# -----------------------------
# 7. Visualisatie
# -----------------------------
low_vis = low.copy()
vis = o3d.visualization.Visualizer()
vis.create_window("3D Bone Viewer", 1200, 900)

vis.add_geometry(geom)
vis.add_geometry(sphere(low_vis, [1, 0, 0]))  # rood
vis.add_geometry(sphere(left, [0, 0, 1]))  # blauw
vis.add_geometry(sphere(tub_pt, [1, 0, 1], 2.5))  # paars
vis.add_geometry(sphere(deep_low, [1, 1, 0]))  # geel
vis.add_geometry(sphere(deep_low_right, [0, 1, 1]))  # cyaan

colors = [[1, 0.5, 0], [0, 1, 0.5]]
for i, pt in enumerate(selected_pts):
    vis.add_geometry(sphere(pt, colors[i]))
if top_bol is not None:
    vis.add_geometry(sphere(top_bol, [1, 0.5, 0]))

opt = vis.get_render_option()
opt.background_color = [0.05, 0.05, 0.05]
opt.light_on = True
opt.mesh_show_back_face = True
opt.point_size = 2
vis.get_view_control().set_zoom(0.8)
vis.run()
vis.destroy_window()
