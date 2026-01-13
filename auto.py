import numpy as np
import open3d as o3d

PLY_FILE = "test7R.ply"  # change if needed


# -------------------------------
# Utility: align geometry to +Y
# -------------------------------
def align_geometry_to_y_axis(geometry):
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        points = np.asarray(geometry.vertices)
    else:
        points = np.asarray(geometry.points)

    centroid = points.mean(axis=0)
    geometry.translate(-centroid)

    cov = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

    if principal_axis[1] < 0:
        principal_axis *= -1

    target_axis = np.array([0.0, 1.0, 0.0])

    v = np.cross(principal_axis, target_axis)
    s = np.linalg.norm(v)
    c = np.dot(principal_axis, target_axis)

    if s < 1e-8:
        return geometry

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    geometry.rotate(R, center=(0, 0, 0))

    return geometry


# -------------------------------
# Load geometry
# -------------------------------
mesh = o3d.io.read_triangle_mesh(PLY_FILE)

if mesh.has_triangles():
    print("Loaded triangle mesh")

    mesh.compute_vertex_normals()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.paint_uniform_color([0.92, 0.92, 0.88])

    geometry = mesh
    points = np.asarray(mesh.vertices)

else:
    print("No triangles found, loading as point cloud")

    pcd = o3d.io.read_point_cloud(PLY_FILE)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
    )

    geometry = pcd
    points = np.asarray(pcd.points)

# -------------------------------
# ALIGN GEOMETRY (face up)
# -------------------------------
geometry = align_geometry_to_y_axis(geometry)

# Re-fetch points after alignment
if isinstance(geometry, o3d.geometry.TriangleMesh):
    points = np.asarray(geometry.vertices)
else:
    points = np.asarray(geometry.points)

# -------------------------------
# Find lowest Y point
# -------------------------------
lowest_index = np.argmin(points[:, 1])
lowest_point = points[lowest_index]
print("Lowest point (Y):", lowest_point)

# -------------------------------
# Rotate bone around Y to face lowest point toward viewer (+Z)
# -------------------------------
vec = lowest_point[[0, 2]]  # x and z
angle = np.arctan2(vec[0], vec[1])  # rotation around Y
R_y = geometry.get_rotation_matrix_from_axis_angle(np.array([0, -angle, 0]))
geometry.rotate(R_y, center=(0, 0, 0))

# Re-fetch points after rotation
if isinstance(geometry, o3d.geometry.TriangleMesh):
    points = np.asarray(geometry.vertices)
else:
    points = np.asarray(geometry.points)

# -------------------------------
# Find "most left" point in bottom 15% of bone
# -------------------------------
Y_min = np.min(points[:, 1])
Y_max = np.max(points[:, 1])
bottom_threshold = Y_min + 0.15 * (Y_max - Y_min)

bottom_points = points[points[:, 1] <= bottom_threshold]

left_index = np.argmin(bottom_points[:, 0])
left_point = bottom_points[left_index]
print("Most left point (bottom 15%):", left_point)

# -------------------------------
# Find highest point along Y-axis line (x≈0, z≈0)
# -------------------------------
axis_tolerance = 1.0  # radius around Y-axis to consider "on axis"

# Distance from Y-axis
dist_from_axis = np.sqrt(points[:, 0] ** 2 + points[:, 2] ** 2)
axis_points = points[dist_from_axis <= axis_tolerance]

if len(axis_points) > 0:
    axis_top_index = np.argmax(axis_points[:, 1])
    axis_top_point = axis_points[axis_top_index]
    print("Top point on Y-axis:", axis_top_point)
else:
    axis_top_point = None
    print("No points found along Y-axis within tolerance!")

# -------------------------------
# Create spheres for landmarks
# -------------------------------
# Red = lowest
sphere_low = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
sphere_low.paint_uniform_color([1.0, 0.0, 0.0])
sphere_low.translate(lowest_point)
sphere_low.compute_vertex_normals()

# Blue = most left (bottom 15%)
sphere_left = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
sphere_left.paint_uniform_color([0.0, 0.0, 1.0])
sphere_left.translate(left_point)
sphere_left.compute_vertex_normals()

# Green = top along Y-axis
if axis_top_point is not None:
    sphere_top = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
    sphere_top.paint_uniform_color([0.0, 1.0, 0.0])
    sphere_top.translate(axis_top_point)
    sphere_top.compute_vertex_normals()




# ============================================================
# EXTRA (AANGEPAST): RADIAL TUBEROSITY — PAPER-CORRECT
# ============================================================

# --- BOVENSTE 8–20% van het bot langs Y-as ---
# We meten vanaf Y_max (proximaal) naar beneden
roi_high = Y_max - 0.08 * (Y_max - Y_min)  # 8% onder de top
roi_low = Y_max - 0.20 * (Y_max - Y_min)  # 20% onder de top

roi_points = points[(points[:, 1] >= roi_low) & (points[:, 1] <= roi_high)]
# Vanaf hier bestaat het bot alleen nog uit het proximale ROI

# --- Schacht-as bepalen in dit gebied ---
# Dit is het "center of the diaphyseal part" uit de tekst
shaft_xz = roi_points[:, [0, 2]].mean(axis=0)  # as in x,z
shaft_y = (roi_low + roi_high) / 2  # midden van ROI
shaft_axis_point = np.array([shaft_xz[0], shaft_y, shaft_xz[1]])

# --- Richting langs het bot (Y-as is al uitgelijnd) ---
bone_axis = np.array([0.0, 1.0, 0.0])


# --- Hoekberekening rond de bot-as ---
# Alles wordt rond de Y-as bekeken, zoals anatomisch logisch
def angular_coords_y(pts, center):
    rel = pts - center
    return np.degrees(np.arctan2(rel[:, 2], rel[:, 0]))


angles = angular_coords_y(roi_points, shaft_axis_point)
# Hier laten we een stuk van 120° met stapjes van 10° om het hele bot heendraaien
# We zoeken het stuk met de meeste punten (grootste volume)
best_mask = None
best_volume = 0

for start in np.arange(-180, 180, 10):   

    mask = (angles >= start) & (angles <= start + 120)
    volume = np.sum(mask)  # punt-aantal ~ volume
    if volume > best_volume:
        best_volume = volume
        best_mask = mask

tuberosity_points = roi_points[best_mask]
# Dit is nu het volledige tuberositas-volume

# --- OPPERVLAKPUNT bepalen (GEEN centroid!) ---
# We zoeken het punt dat het verst van de schacht-as ligt
rel = tuberosity_points - shaft_axis_point
radial_dist = np.sqrt(rel[:, 0] ** 2 + rel[:, 2] ** 2)

surface_point = tuberosity_points[np.argmax(radial_dist)]
# Dit punt ligt gegarandeerd op de bolling

# --- Paarse sphere op correcte tuberositas-positie ---
sphere_tuberosity = o3d.geometry.TriangleMesh.create_sphere(radius=2.5)
sphere_tuberosity.paint_uniform_color([1.0, 0.0, 1.0])  # PAARS
sphere_tuberosity.translate(surface_point)
sphere_tuberosity.compute_vertex_normals()


# -------------------------------
# Visualization
# -------------------------------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Bone Viewer", width=1200, height=900)

vis.add_geometry(geometry)
vis.add_geometry(sphere_low)
vis.add_geometry(sphere_left)
vis.add_geometry(sphere_tuberosity)
if axis_top_point is not None:
    vis.add_geometry(sphere_top)

opt = vis.get_render_option()
opt.background_color = np.array([0.05, 0.05, 0.05])
opt.light_on = True
opt.mesh_show_back_face = True
opt.point_size = 2.0

ctr = vis.get_view_control()
ctr.set_zoom(0.8)

vis.run()
vis.destroy_window()

