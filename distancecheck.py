import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# =====================
# Instellingen
# =====================
PLY_FILE = "bot270.ply"
MANUAL_PLY = "landmarks270.ply"
BONE_LENGTH_CM = 29.1
MIN_DIST_GEEL_CYAAN = 12.0

# =====================
# Hulpfuncties
# =====================
def pts(g):
    return np.asarray(g.vertices if isinstance(g, o3d.geometry.TriangleMesh) else g.points)

def sphere(p, c, r=2.5):
    s = o3d.geometry.TriangleMesh.create_sphere(r)
    s.paint_uniform_color(c)
    s.translate(p)
    s.compute_vertex_normals()
    return s

def apply_transform(points, T):
    pts_h = np.c_[points, np.ones(len(points))]
    return (T @ pts_h.T).T[:, :3]

def fit_sphere(points):
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    A = np.c_[2*X, 2*Y, 2*Z, np.ones(len(X))]
    f = X**2 + Y**2 + Z**2
    C, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
    center = C[:3]
    radius = np.sqrt(C[3] + np.sum(center**2))
    rmse = np.sqrt(np.mean((np.linalg.norm(points - center, axis=1) - radius)**2))
    return center, radius, rmse

# =====================
# 1. Inlezen
# =====================
geom = o3d.io.read_triangle_mesh(PLY_FILE)
geom.compute_vertex_normals()
geom.remove_duplicated_vertices()
geom.remove_degenerate_triangles()
geom.paint_uniform_color([0.92, 0.92, 0.88])

manual_pc = o3d.io.read_point_cloud(MANUAL_PLY)
manual_pts = np.asarray(manual_pc.points)

p = pts(geom)

# =====================
# 2. Consistente oriëntatie (zoals je eerste script)
# =====================
center = p.mean(axis=0)
p0 = p - center
cov = np.cov(p0.T)
_, v = np.linalg.eigh(cov)
axis = v[:, -1]
if axis[1] < 0:
    axis *= -1

cross = np.cross(axis, [0,1,0])
s = np.linalg.norm(cross)

R = np.eye(3)
if s > 1e-8:
    vx = np.array([[0,-cross[2],cross[1]],[cross[2],0,-cross[0]],[-cross[1],cross[0],0]])
    R = np.eye(3) + vx + vx@vx*((1-axis[1])/s**2)

T = np.eye(4)
T[:3,:3] = R
T[:3,3] = -R @ center

geom.transform(T)
manual_pts = apply_transform(manual_pts, T)
p = pts(geom)

# =====================
# 3. Onderkant naar beneden + rotatie rond Y
# =====================
ymin, ymax = p[:,1].min(), p[:,1].max()
h = ymax - ymin

bottom = p[p[:,1] <= ymin + 0.2*h]
top = p[p[:,1] >= ymax - 0.2*h]
if np.std(bottom[:,[0,2]]) < np.std(top[:,[0,2]]):
    Rf = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi,0,0])
    Tf = np.eye(4)
    Tf[:3,:3] = Rf
    geom.transform(Tf)
    manual_pts = apply_transform(manual_pts, Tf)
    p = pts(geom)

low = p[np.argmin(p[:,1])]
angle = np.arctan2(low[0], low[2])
Ry = o3d.geometry.get_rotation_matrix_from_axis_angle([0,-angle,0])
Ty = np.eye(4)
Ty[:3,:3] = Ry
geom.transform(Ty)
manual_pts = apply_transform(manual_pts, Ty)
p = pts(geom)

ymin, ymax = p[:,1].min(), p[:,1].max()
zmin, zmax = p[:,2].min(), p[:,2].max()
x_min, x_max = p[:,0].min(), p[:,0].max()
h = ymax - ymin

# =====================
# 4. Automatische landmarks
# =====================
flip = False
for _ in range(2):
    p_m = p.copy()
    if flip:
        p_m[:,0] *= -1

    y_norm = (p_m[:,1]-ymin)/h
    z_norm = (p_m[:,2]-zmin)/(zmax-zmin)

    deep_lr = p_m[np.argmin(0.8*y_norm+0.2*z_norm)]

    bottom_roi = p_m[p_m[:,1] <= ymin + 0.15*h]
    left = bottom_roi[np.argmin((bottom_roi[:,0]-x_min)/(x_max-x_min))]

    roi = p_m[(p_m[:,1] >= ymax-0.20*h) & (p_m[:,1] <= ymax-0.08*h)]
    shaft_xz = roi[:,[0,2]].mean(axis=0)
    shaft = np.array([shaft_xz[0], roi[:,1].mean(), shaft_xz[1]])

    angles = np.degrees(np.arctan2(roi[:,2]-shaft[2], roi[:,0]-shaft[0]))
    mask = (angles>-60) & (angles<60)
    tub_pt = roi[mask][np.argmax(np.linalg.norm(roi[mask][:,[0,2]]-shaft[[0,2]],axis=1))]

    deep_low = p_m[np.argmin(0.8*y_norm + 0.2*z_norm)]
    deep_low_right = p_m[np.argmin(0.6*y_norm + 0.15*z_norm - 0.25*((p_m[:,0]-x_min)/(x_max-x_min)))]

    if np.linalg.norm(deep_low - deep_low_right) >= MIN_DIST_GEEL_CYAAN:
        break
    flip = not flip

if flip:
    left[0] *= -1
    tub_pt[0] *= -1
    deep_low[0] *= -1
    deep_low_right[0] *= -1

# =====================
# 5. Kuilen onderaan (zoals in je laatste werkende code)
# =====================
vertices = np.asarray(geom.vertices)
roi_fraction = 0.07
neighborhood_radius = 6
n_kuilen = 2
min_dist = 8

y_limit = ymin + roi_fraction*h
roi_mask = vertices[:,1] <= y_limit
roi_points = vertices[roi_mask][::5]

xz_center = roi_points[:,[0,2]].mean(axis=0)
bottom_mask = np.linalg.norm(roi_points[:,[0,2]]-xz_center,axis=1) <= 0.55*np.max(np.linalg.norm(roi_points[:,[0,2]]-xz_center,axis=1))
roi_points = roi_points[bottom_mask]

tree = cKDTree(roi_points[:,[0,2]])
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
    if all(np.linalg.norm(pt[[0,2]] - p[[0,2]]) >= min_dist for p in selected_pts):
        _, idx_mesh = cKDTree(p[:,[0,2]]).query([pt[0], pt[2]])
        selected_pts.append(np.array([pt[0], p[idx_mesh,1], pt[2]]))
    if len(selected_pts) >= n_kuilen:
        break

# =====================
# 6. Top bol
# =====================
roi_fraction_top = 0.05
y_limit_top = ymax - roi_fraction_top*h
top_points = vertices[vertices[:,1]>=y_limit_top][::5]

tree_top = cKDTree(top_points[:,[0,2]])
sphere_scores_top = []
for pt in top_points:
    idx = tree_top.query_ball_point([pt[0], pt[2]], neighborhood_radius)
    if len(idx)<5:
        continue
    _, _, rmse = fit_sphere(top_points[idx])
    sphere_scores_top.append((pt, rmse))

top_bol = min(sphere_scores_top, key=lambda x:x[1])[0] if sphere_scores_top else None

# =====================
# 7. Foutberekening
# =====================
auto_landmarks = {
    "rood (laagste punt)": low,
    "blauw (Lister)": left,
    "paars (Tuberosity)": tub_pt,
    "geel (Dorsal rim)": deep_low,
    "cyaan (Volar rim)": deep_low_right,
    "oranje (Kuil 1)": selected_pts[0],
    "groen (Kuil 2)": selected_pts[1],
}

if top_bol is not None:
    auto_landmarks["wit (Top bol)"] = top_bol

cm_per_unit = BONE_LENGTH_CM/h
tree_manual = cKDTree(manual_pts)

print("\n--- FOUT PER LANDMARK (cm) ---")
for name, pt in auto_landmarks.items():
    dist_units,_ = tree_manual.query(pt)
    print(f"{name:<28} : {dist_units*cm_per_unit:6.2f} cm")

# =====================
# 8. Visualisatie
# =====================
vis = o3d.visualization.Visualizer()
vis.create_window("3D Bone Viewer", 1200,900)
vis.add_geometry(geom)

colors = {
    "rood (laagste punt)": [1,0,0],
    "blauw (Lister)": [0,0,1],
    "paars (Tuberosity)": [1,0,1],
    "geel (Dorsal rim)": [1,1,0],
    "cyaan (Volar rim)": [0,1,1],
    "oranje (Kuil 1)": [1,0.5,0],
    "groen (Kuil 2)": [0,1,0.5],
    "wit (Top bol)": [1,1,1],
}

for name, pt in auto_landmarks.items():
    vis.add_geometry(sphere(pt, colors[name]))

opt = vis.get_render_option()
opt.background_color = [0.05,0.05,0.05]
opt.mesh_show_back_face = True
vis.get_view_control().set_zoom(0.8)
vis.run()
vis.destroy_window()
