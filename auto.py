import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

PLY_FILE = "test4.ply"  # bestand met 3D botmodel


# Geef punten terug
def pts(g):
    return np.asarray(
        g.vertices if isinstance(g, o3d.geometry.TriangleMesh) else g.points
    )


# Maak een bol op positie p, met kleur c en radius r.
def sphere(p, c, r=2.0):
    s = o3d.geometry.TriangleMesh.create_sphere(r)
    s.paint_uniform_color(c)
    s.translate(p)
    s.compute_vertex_normals()
    return s


# Zet het bot recht met +y omhoog.
def align_to_y(g):
    p = pts(g)
    g.translate(-p.mean(axis=0))  # centrum bot op oorsprong
    _, v = np.linalg.eigh(np.cov(p.T))  # richtingen bepalen
    axis = v[:, -1]
    if axis[1] < 0:
        axis *= -1  # omhoog gericht
    cross = np.cross(axis, [0, 1, 0])
    s = np.linalg.norm(cross)
    if s < 1e-8:  # als s kleiner, = goed georiënteerd
        return g
    # Rotatiematrix rond y.
    vx = np.array(
        [[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]]
    )
    R = np.eye(3) + vx + vx @ vx * ((1 - axis[1]) / s**2)
    g.rotate(R, center=(0, 0, 0))
    return g


geom = o3d.io.read_triangle_mesh(PLY_FILE)
if geom.has_triangles():
    # Mesh opschonen
    geom.compute_vertex_normals()
    geom.remove_duplicated_vertices()
    geom.remove_degenerate_triangles()
    geom.paint_uniform_color([0.92, 0.92, 0.88])
else:
    # Point cloud
    geom = o3d.io.read_point_cloud(PLY_FILE)
    geom.estimate_normals()

# Uitlijnen en punten ophalen.
geom = align_to_y(geom)
p = pts(geom)

# Laagste punt vinden.
low = p[np.argmin(p[:, 1])]

# Rotatie zodat laagste punt naar +z wijst.
angle = np.arctan2(low[0], low[2])
geom.rotate(geom.get_rotation_matrix_from_axis_angle([0, -angle, 0]), center=(0, 0, 0))
p = pts(geom)

# Basis grenzen
ymin, ymax = p[:, 1].min(), p[:, 1].max()
zmin, zmax = p[:, 2].min(), p[:, 2].max()
x_min, x_max = p[:, 0].min(), p[:, 0].max()
h = ymax - ymin

# LANDMARKS
# Lister’s tubercle (blauw).
# Laagste 15% ROI , min() van x-as.
bottom = p[p[:, 1] <= ymin + 0.15 * h]
left = bottom[np.argmin(bottom[:, 0])]

# Center of radial head/center of humeral fossa (groen).
# Hoogste punt direct op de y-as.
axis_pts = p[np.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) <= 1.0]
top_axis = axis_pts[np.argmax(axis_pts[:, 1])] if len(axis_pts) else None

# Radial tuberosity (paars).
# Vind het punt door een ROI te selecteren, het middelpunt van de schacht te bepalen, een 120° sector met de meeste punten te kiezen en vervolgens het verste punt van die sector te nemen.
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

# Peak of dorsal rim on sigmoid notch (geel).
# Ratio 80/20 van min() op de y-as en z-as om punt te vinden.
y_norm = (p[:, 1] - ymin) / h
z_norm = (p[:, 2] - zmin) / (zmax - zmin)
score = 0.8 * y_norm + 0.2 * z_norm
deep_low = p[np.argmin(score)]

# Peak of volar rim on sigmoid notch (cyaan).
# Ratio 60/25/15 van min() op de y-as, z-as en max() van x-as om punt te vinden.
x_norm = (p[:, 0] - x_min) / (x_max - x_min)
score_xyz = 0.6 * y_norm + 0.25 * z_norm - 0.15 * x_norm
deep_low_right = p[np.argmin(score_xyz)]

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# Kuiltjes (dimples) detection
roi_fraction = 0.10  # onderste 10% van het bot
neighborhood_radius = 5  # radius van buurtpunten voor bolfit
n_kuilen = 2  # aantal kuilen om terug te geven
min_dist = 5.0  # minimale afstand tussen de spheres

# --- 1. ROI: onderste 10% ---
ymin = p[:, 1].min()
ymax = p[:, 1].max()
h = ymax - ymin
y_limit = ymin + roi_fraction * h
roi_points = p[p[:, 1] <= y_limit]

print(f"Aantal punten in ROI: {len(roi_points)}")

# --- 2. KD-tree voor snelle buurtzoeking ---
tree = cKDTree(roi_points[:, [0, 2]])


# --- 3. Functie voor bolfit in 3D ---
def fit_sphere(points):
    # algebraic fit: ||X-C||^2 = r^2
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    A = np.c_[2 * X, 2 * Y, 2 * Z, np.ones(len(X))]
    f = X**2 + Y**2 + Z**2
    C, residuals, _, _ = np.linalg.lstsq(A, f, rcond=None)
    center = C[:3]
    radius = np.sqrt(C[3] + np.sum(center**2))
    # RMSE van de fit
    rmse = np.sqrt(np.mean((np.linalg.norm(points - center, axis=1) - radius) ** 2))
    return center, radius, rmse


# --- 4. Bolfit over alle punten in ROI ---
sphere_scores = []
for i, pt in enumerate(roi_points):
    idx = tree.query_ball_point([pt[0], pt[2]], neighborhood_radius)
    if len(idx) < 5:  # niet genoeg punten om te fitten
        continue
    local_points = roi_points[idx]
    center, radius, rmse = fit_sphere(local_points)
    sphere_scores.append((pt, rmse))

sphere_scores.sort(key=lambda x: x[1])  # kleinste RMSE eerst
selected_pts = []

# --- 5. Kies n_kuilen met minimale afstand ---
for pt, score in sphere_scores:
    if all(np.linalg.norm(pt[[0, 2]] - p[[0, 2]]) >= min_dist for p in selected_pts):
        # pas Y aan zodat de sphere exact op het oppervlak zit
        _, idx_mesh = cKDTree(p[:, [0, 2]]).query([pt[0], pt[2]])
        surface_pt = np.array([pt[0], p[idx_mesh, 1], pt[2]])
        selected_pts.append(surface_pt)
    if len(selected_pts) >= n_kuilen:
        break

print("Geselecteerde kuilen via bolfit:", selected_pts)
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Visualization


# Visualisatie
vis = o3d.visualization.Visualizer()
vis.create_window("3D Bone Viewer", 1200, 900)

# Voeg geometrie en punten toe.
vis.add_geometry(geom)
vis.add_geometry(sphere(low, [1, 0, 0]))  # rood
vis.add_geometry(sphere(left, [0, 0, 1]))  # blauw
vis.add_geometry(sphere(tub_pt, [1, 0, 1], 2.5))  # paars
vis.add_geometry(sphere(deep_low, [1, 1, 0]))  # geel
vis.add_geometry(sphere(deep_low_right, [0, 1, 1]))  # cyaan
if top_axis is not None:
    vis.add_geometry(sphere(top_axis, [0, 1, 0]))  # groen
colors = [[1,0.5,0],[0,1,0.5]]  # max 2 kuiltjes
for i, pt in enumerate(selected_pts):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=2.5)
    s.paint_uniform_color(colors[i])
    s.translate(pt)
    s.compute_vertex_normals()
    vis.add_geometry(s)    

# Render opties
opt = vis.get_render_option()
opt.background_color = [0.05, 0.05, 0.05]
opt.light_on = True
opt.mesh_show_back_face = True
opt.point_size = 2
vis.get_view_control().set_zoom(0.8)

vis.run()
vis.destroy_window()



