import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

PLY_FILE = "15L.ply"
MIN_DIST_GEEL_CYAAN = 12.0


# =====================
# Hulpfuncties
# =====================
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

    cov = np.cov(p.T)
    if not np.isfinite(cov).all():
        return g

    _, v = np.linalg.eigh(cov)
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
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    A = np.c_[2 * X, 2 * Y, 2 * Z, np.ones(len(X))]
    f = X**2 + Y**2 + Z**2
    C, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
    center = C[:3]
    radius = np.sqrt(C[3] + np.sum(center**2))
    rmse = np.sqrt(np.mean((np.linalg.norm(points - center, axis=1) - radius) ** 2))
    return center, radius, rmse


# =====================
# 1. Inlezen
# =====================
geom = o3d.io.read_triangle_mesh(PLY_FILE)
geom.compute_vertex_normals()
geom.remove_duplicated_vertices()
geom.remove_degenerate_triangles()
geom.paint_uniform_color([0.92, 0.92, 0.88])

geom = align_to_y(geom)
p = pts(geom)

ymin, ymax = p[:, 1].min(), p[:, 1].max()
h = ymax - ymin

bottom = p[p[:, 1] <= ymin + 0.20 * h]
top = p[p[:, 1] >= ymax - 0.20 * h]
if np.std(bottom[:, [0, 2]]) < np.std(top[:, [0, 2]]):
    geom.rotate(
        geom.get_rotation_matrix_from_axis_angle([np.pi, 0, 0]), center=(0, 0, 0)
    )
    p = pts(geom)

low = p[np.argmin(p[:, 1])]
angle = np.arctan2(low[0], low[2])
geom.rotate(geom.get_rotation_matrix_from_axis_angle([0, -angle, 0]), center=(0, 0, 0))
p = pts(geom)

ymin, ymax = p[:, 1].min(), p[:, 1].max()
zmin, zmax = p[:, 2].min(), p[:, 2].max()
x_min, x_max = p[:, 0].min(), p[:, 0].max()
h = ymax - ymin


# =====================
# 2–5. Links/rechts + landmarks (retry met spiegelen)
# =====================
flip = False

for _ in range(2):
    p_mirror = p.copy()
    if flip:
        p_mirror[:, 0] *= -1

    y_norm = (p_mirror[:, 1] - ymin) / h
    z_norm = (p_mirror[:, 2] - zmin) / (zmax - zmin)
    score_full = 0.8 * y_norm + 0.2 * z_norm
    deep_lr = p_mirror[np.argmin(score_full)]
    left_arm = deep_lr[0] < 0

    bottom_roi = p_mirror[p_mirror[:, 1] <= ymin + 0.15 * h]
    score_blue = 0.8 * (bottom_roi[:, 0] - x_min) / (x_max - x_min) + 0.2 * (
        bottom_roi[:, 2] - zmin
    ) / (zmax - zmin)
    left = bottom_roi[np.argmin(score_blue)]

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

    score_yz = 0.8 * y_norm + 0.2 * z_norm
    deep_low = p_mirror[np.argmin(score_yz)]

    x_norm = (p_mirror[:, 0] - x_min) / (x_max - x_min)
    score_xyz = 0.6 * y_norm + 0.15 * z_norm - 0.25 * x_norm
    deep_low_right = p_mirror[np.argmin(score_xyz)]

    if np.linalg.norm(deep_low - deep_low_right) >= MIN_DIST_GEEL_CYAAN:
        break

    flip = not flip


# =====================
# 5. Terug spiegelen (exact één keer)
# =====================
if flip:
    left[0] *= -1
    tub_pt[0] *= -1
    deep_low[0] *= -1
    deep_low_right[0] *= -1


# =====================
# 6. Kuilen onderaan
# =====================
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
for pt in roi_points:
    idx = tree.query_ball_point([pt[0], pt[2]], neighborhood_radius)
    if len(idx) < 5:
        continue
    center, radius, rmse = fit_sphere(roi_points[idx])
    sphere_scores.append((pt, rmse))

sphere_scores.sort(key=lambda x: x[1])

selected_pts = []
for pt, _ in sphere_scores:
    if all(np.linalg.norm(pt[[0, 2]] - p[[0, 2]]) >= min_dist for p in selected_pts):
        _, idx_mesh = cKDTree(p[:, [0, 2]]).query([pt[0], pt[2]])
        selected_pts.append(np.array([pt[0], p[idx_mesh, 1], pt[2]]))
    if len(selected_pts) >= n_kuilen:
        break


# =====================
# 7. Top bol
# =====================
roi_fraction_top = 0.05
y_limit_top = ymax - roi_fraction_top * h
top_points = vertices[vertices[:, 1] >= y_limit_top][::5]

tree_top = cKDTree(top_points[:, [0, 2]])
sphere_scores_top = []
for pt in top_points:
    idx = tree_top.query_ball_point([pt[0], pt[2]], neighborhood_radius)
    if len(idx) < 5:
        continue
    _, _, rmse = fit_sphere(top_points[idx])
    sphere_scores_top.append((pt, rmse))

top_bol = min(sphere_scores_top, key=lambda x: x[1])[0] if sphere_scores_top else None


# =====================
# 8. Visualisatie
# =====================
vis = o3d.visualization.Visualizer()
vis.create_window("3D Bone Viewer", 1200, 900)
vis.add_geometry(geom)

vis.add_geometry(sphere(low, [1, 0, 0]))
vis.add_geometry(sphere(left, [0, 0, 1]))
vis.add_geometry(sphere(tub_pt, [1, 0, 1]))
vis.add_geometry(sphere(deep_low, [1, 1, 0]))
vis.add_geometry(sphere(deep_low_right, [0, 1, 1]))

colors = [[1, 0.5, 0], [0, 1, 0.5]]
for i, pt in enumerate(selected_pts):
    vis.add_geometry(sphere(pt, colors[i]))

if top_bol is not None:
    vis.add_geometry(sphere(top_bol, [1, 0.5, 0]))

opt = vis.get_render_option()
opt.background_color = [0.05, 0.05, 0.05]
opt.mesh_show_back_face = True
vis.get_view_control().set_zoom(0.8)

vis.run()
vis.destroy_window()
