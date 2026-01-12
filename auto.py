import numpy as np
import open3d as o3d

PLY_FILE = "test1.ply"  # change if needed


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
# Vector from origin to lowest point in XZ plane
vec = lowest_point[[0, 2]]  # x and z
angle = np.arctan2(vec[0], vec[1])  # angle around Y to rotate

# Rotate around Y (up) by -angle
R_y = geometry.get_rotation_matrix_from_axis_angle(np.array([0, -angle, 0]))
geometry.rotate(R_y, center=(0, 0, 0))

# -------------------------------
# Create big point (sphere)
# -------------------------------
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=4.0)
sphere.paint_uniform_color([1.0, 0.0, 0.0])  # red
sphere.translate(lowest_point)
sphere.compute_vertex_normals()

# -------------------------------
# Visualization
# -------------------------------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Bone Viewer", width=1200, height=900)

vis.add_geometry(geometry)
vis.add_geometry(sphere)

opt = vis.get_render_option()
opt.background_color = np.array([0.05, 0.05, 0.05])
opt.light_on = True
opt.mesh_show_back_face = True
opt.point_size = 2.0

ctr = vis.get_view_control()
ctr.set_zoom(0.8)

vis.run()
vis.destroy_window()
