import numpy as np
import open3d as o3d

PLY_FILE = "test.ply"  # change if needed


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

# -------------------------------
# Visualization
# -------------------------------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Bone Viewer", width=1200, height=900)

vis.add_geometry(geometry)
vis.add_geometry(sphere_low)
vis.add_geometry(sphere_left)
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
