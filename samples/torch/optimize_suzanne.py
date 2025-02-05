from utils.smooth_template_fixed_light import render_with_lighting as render_with_lighting_sphere, compute_vertex_normals, compute_face_normals, rotate_y, load_obj
import argparse
import os
import pathlib
import numpy as np
import torch
import imageio
import util
import nvdiffrast.torch as dr
import torch.nn.functional as F
from gpytoolbox import remesh_botsch, non_manifold_edges

def compute_edge_length_loss(vertices, edges, reference_lengths):
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    current_lengths = torch.norm(v0 - v1, dim=1)
    return torch.mean((current_lengths - reference_lengths) ** 2)

def compute_face_based_reference_lengths(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1_ref = torch.norm(v0 - v1, dim=1)
    edge2_ref = torch.norm(v1 - v2, dim=1)
    edge3_ref = torch.norm(v2 - v0, dim=1)
    # Stack them so that the shape is (num_faces, 3).
    return torch.stack([edge1_ref, edge2_ref, edge3_ref], dim=1)

def compute_edge_lengths(vertices, edges):
    """
    Compute reference edge lengths based on the original mesh configuration,
    given an array of edges. Each edge is defined by two vertex indices.
    """
    # Get vertex positions for each edge
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]

    # Compute the length of each edge
    lengths = torch.norm(v0 - v1, dim=1)

    return lengths

def apply_lighting(vertices, normals, interpolated_light_dirs, light_colors, ambient_color, view_position):
    # Expand ambient_color to match the shape of the normals
    ambient_color = ambient_color[None, None, None, :].expand_as(normals).clone()

    # Initialize the shading with ambient lighting
    shading = ambient_color.clone()

    # Loop over each light source and accumulate its contribution
    for light_dir, light_color in zip(interpolated_light_dirs, light_colors):
        # Compute light direction
        light_dir = F.normalize(light_dir, dim=-1)

        # Compute the diffuse lighting (Lambertian model)
        diffuse_intensity = torch.clamp(torch.sum(normals * light_dir, dim=-1, keepdim=True), 0.0, 1.0)
        diffuse = diffuse_intensity * light_color

        # Accumulate the diffuse contribution from each light
        shading += diffuse

    #####################################################################################################

    # ambient_color = ambient_color[None, None, None, :].expand_as(normals).clone()

    # Initialize the shading with ambient lighting
    # shading = ambient_color.clone()

    # # Compute a global light contribution (hemispherical light model)
    # diffuse_intensity = torch.clamp(normals[..., 2:3], 0.0, 1.0)  # Z+ normals receive more light
    # global_light = diffuse_intensity * ambient_color  # Apply ambient light intensity based on normal direction

    # # Accumulate global contribution
    # shading += global_light

    return shading

def render_with_lighting(glctx, mvp_batch, pos, pos_idx, vtx_col, col_idx, vtx_normals, normal_idx, resolution, light_positions=None, light_colors=None, ambient_color=None):
    """
    Render with lighting for a batch of MVP matrices.

    Args:
        glctx (RasterizeCudaContext): Context for nvdiffrast.
        mvp_batch (torch.Tensor): Batched MVP matrices of shape (N_view, 4, 4).
        pos (torch.Tensor): Vertex positions of shape (N_vertices, 3).
        pos_idx (torch.Tensor): Triangle indices of shape (N_triangles, 3).
        vtx_col (torch.Tensor): Vertex colors of shape (N_vertices, 3).
        col_idx (torch.Tensor): Vertex color indices of shape (N_triangles, 3).
        vtx_normals (torch.Tensor): Vertex normals of shape (N_vertices, 3).
        normal_idx (torch.Tensor): Vertex normal indices of shape (N_triangles, 3).
        resolution (int): Output image resolution.
        light_positions (list[torch.Tensor], optional): List of light positions. Defaults to None.
        light_colors (list[torch.Tensor], optional): List of light colors corresponding to the positions. Defaults to None.
        ambient_color (list[float]): RGB values for ambient light.

    Returns:
        shaded_color_batch (torch.Tensor): Rendered colors for all views, shape (N_view, 3, H, W).
        rast_out_batch (torch.Tensor): Rasterization outputs for all views.
    """
    N_view = mvp_batch.shape[0]

    # Scale and transform positions for all views
    pos_scaled = pos / 1
    pos_homo = torch.cat([pos_scaled, torch.ones_like(pos_scaled[:, :1])], dim=-1)  # Add w-component
    pos_clip_batch = torch.matmul(pos_homo, mvp_batch.transpose(1, 2))  # Shape: (N_view, N_vertices, 4)

    rast_out_batch = []
    shaded_color_batch = []

    for i in range(N_view):
        # Rasterize for each view
        pos_clip = pos_clip_batch[i]  # Shape: [N_vertices, 4]

        # Define ranges explicitly and move to CPU
        ranges = torch.tensor([[0, pos_idx.shape[0]]], dtype=torch.int32).to('cpu')  # Use all triangles

        rast_out, _ = dr.rasterize(
            glctx,
            pos_clip.contiguous(),
            pos_idx.contiguous(),
            resolution=[resolution, resolution],
            ranges=ranges  # Explicitly pass ranges (on CPU)
        )
        rast_out_batch.append(rast_out)

        # Interpolate attributes for each view
        color, _ = dr.interpolate(vtx_col[None, ...].contiguous(), rast_out, col_idx.contiguous())
        normal, _ = dr.interpolate(vtx_normals[None, ...].contiguous(), rast_out, normal_idx.contiguous())

        # Antialiasing
        color = dr.antialias(color, rast_out, pos_clip, pos_idx.contiguous())

        # Compute light directions if light sources are provided
        if len(light_positions) > 0 and len(light_colors) > 0:
            interpolated_light_dirs = []
            for light_position in light_positions:
                light_dir = light_position - pos[None, ...]
                light_dir = F.normalize(light_dir, dim=-1)
                light_dir_interpolated, _ = dr.interpolate(light_dir.contiguous(), rast_out, pos_idx.contiguous())
                interpolated_light_dirs.append(light_dir_interpolated)

            # Apply lighting
            view_position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
            ambient_color_tensor = torch.tensor(ambient_color, dtype=torch.float32, device='cuda')
            shaded_color = apply_lighting(
                pos[None, ...], normal, interpolated_light_dirs, light_colors, ambient_color_tensor, view_position
            )
        else:
            # Only ambient light
            shaded_color = torch.tensor(ambient_color, dtype=torch.float32, device='cuda')

        shaded_color = color * shaded_color  # Multiply with interpolated color

        # Append the result (remove the extra dimension)
        shaded_color_batch.append(shaded_color.squeeze(0))  # Remove batch dimension for single view

    # Stack the results into a batch
    shaded_color_batch = torch.stack(shaded_color_batch, dim=0)  # (N_view, 3, H, W)
    rast_out_batch = torch.stack(rast_out_batch, dim=0)  # (N_view, H, W, ...)

    return shaded_color_batch, rast_out_batch

def make_grid(arr, ncols=2):
    n, height, width, nc = arr.shape
    nrows = (n + ncols - 1) // ncols  # Compute the number of rows (ceiling division)
    
    # Create an empty canvas for the grid
    grid = np.zeros((nrows * height, ncols * width, nc), dtype=arr.dtype)
    
    for idx, img in enumerate(arr):
        row = idx // ncols
        col = idx % ncols
        grid[row * height:(row + 1) * height, col * width:(col + 1) * width, :] = img
    
    return grid

def rotate_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def set_rotation_translation(x_angle=0.0, y_angle=0.0, z_angle=0.0, translation=(0.0, 0.0, 0.0)):
    """
    Set rotation and translation with user-specified values.
    
    Args:
        x_angle (float): Rotation angle around the X-axis in radians.
        y_angle (float): Rotation angle around the Y-axis in radians.
        z_angle (float): Rotation angle around the Z-axis in radians.
        translation (tuple of 3 floats): Translation offsets (x, y, z).
    
    Returns:
        numpy.ndarray: A 4x4 transformation matrix with the specified rotation and translation.
    """
    # Generate rotation matrices for each axis
    rot_x = util.rotate_x(x_angle)
    rot_y = util.rotate_y(y_angle)
    rot_z = rotate_z(z_angle)
    
    # Combine rotations in the order: Z -> Y -> X
    rotation_matrix = np.matmul(rot_z, np.matmul(rot_y, rot_x))
    
    # Add translation
    transformation_matrix = np.matmul(util.translate(*translation), rotation_matrix)
    
    return transformation_matrix

def optimize_cube_to_sphere(max_iter=5000, resolution=512, display_res=512, out_dir=None):
    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'
    obj_file = f'{datadir}/sphere_ico_high_4.obj'

    # Load vertices and faces from the OBJ file
    vertices_opt, faces_opt = load_obj(obj_file)
    print(f"vertices type: {type(vertices_opt)}")
    # vtx_pos, pos_idx = load_obj(obj_file)

    # print("1 Mesh cube has %d triangles shape." % (faces_opt[0].shape[0]))

    print("Mesh sphere has %d triangles and %d vertices." % (faces_opt.shape[0], vertices_opt.shape[0]))

    # Subdivision intervals 
    subdiv_iters = [10]

    # SPHERE
    # Compute face normals
    face_normals_opt = compute_face_normals(vertices_opt, faces_opt)
    vtx_normals_opt = compute_vertex_normals(vertices_opt,  faces_opt, face_normals_opt)

    vtx_col_opt = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda').expand(vertices_opt.shape[0], -1)

    col_idx_opt = faces_opt    # Color indices match position indices
    normal_idx_opt = faces_opt    # Normal indices match position indices
    ###################################################################################################

    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'
    obj_file = f'{datadir}/suzanne.obj'

    # Load vertices and faces from the OBJ file
    vertices, faces = load_obj(obj_file)
    
    print(f"Mesh suzanne has vertices type: {type(vertices)}")
    print("Mesh suzanne has %d triangles and %d vertices." % (faces.shape[0], vertices.shape[0]))

    # # Compute face normals
    face_normals = compute_face_normals(vertices, faces)
    vtx_normals = compute_vertex_normals(vertices, faces, face_normals)

    # vertices = inflate_to_sphere(vertices, radius=1.0)
    vtx_col = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda').expand(vertices.shape[0], -1)

    col_idx = faces    # Color indices match position indices
    normal_idx = faces  # Normal indices match position indices
    ####################################################################################################

    glctx = dr.RasterizeCudaContext()

    # Reference light parameters (positions and colors)
    reference_light_positions = torch.tensor([
        [-5.0, 5.0, -5.0],
        [5.0, -5.0, 5.0]
    ], dtype=torch.float32, device='cuda')

    reference_light_colors = torch.tensor([
        [1.0, 0.0, 0.0],  
        [0.0, 1.0, 0.0],  
        # [0.0, 1.0, 0.0],  
        # [1.0, 0.0, 0.0]   
    ], dtype=torch.float32, device='cuda')

    sphere_light_positions = torch.tensor([
        [-5.0, 5.0, -5.0],  
        [5.0, -5.0, 5.0]
        # [-7.0, -7.0, 0.0],  
        # [0.0, 0.0, 4.0]    
    ], dtype=torch.float32, device='cuda')

    sphere_light_colors = torch.tensor([
        [1.0, 0.0, 0.0],  
        [0.0, 1.0, 0.0],  
        # [0.0, 1.0, 0.0],  
        # [1.0, 0.0, 0.0]   
    ], dtype=torch.float32, device='cuda')

    # reference_light_positions = []
    # reference_light_colors = []
    # sphere_light_positions = []
    # sphere_light_colors = []


    ambient_color = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32, device='cuda')

    ####################################################################################################

    ang = 0.0

    log_file = None
    writer = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        log_file = open(os.path.join(out_dir, 'log.txt'), 'w')
        writer = imageio.get_writer(os.path.join(out_dir, 'sphere_to_suzanne.mp4'), mode='I', fps=30, codec='libx264', bitrate='16M')

    log_interval = 1
    display_interval = 1
    mp4save_interval = 1

    ####################################################################################################

    # After all desired subdivisions are done, enable gradient tracking:
    vertices_opt.requires_grad_(True)
    lr = 1e-2
    # # Now define the optimizer and scheduler
    optimizer = torch.optim.Adam([vertices_opt], lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.01, 10 ** (-x * 0.0005)))

    ang = 0.0
    gl_avg = []

    # Extract unique edges from faces_opt
    edges_set = set()
    f_cpu = faces_opt.cpu().numpy()
    for tri in f_cpu:
        tri = tri.tolist()
        edges_set.add(tuple(sorted((tri[0], tri[1]))))
        edges_set.add(tuple(sorted((tri[1], tri[2]))))
        edges_set.add(tuple(sorted((tri[2], tri[0]))))

    edges_list = list(edges_set)
    # print(edges_list)
    edges_tensor = torch.tensor(edges_list, dtype=torch.long, device='cuda') 

    # print(f"edge tensor shape: {edges_tensor.shape}")

    reference_edge_lengths = compute_edge_lengths(vertices_opt, edges_tensor).detach()

    # Weight of edge length regularization
    lambda_edge = 1.0
    for it in range(max_iter + 1):

        if it in subdiv_iters:
            # Check face indices are within bounds
            max_index = faces_opt.max()
            num_vertices = vertices_opt.shape[0]

            if max_index >= num_vertices:
                raise ValueError(f"Face index out-of-bounds! Max index: {max_index}, Number of vertices: {num_vertices}")

            with torch.no_grad():

                vtx_pos_remeshed, pos_idx_remeshed = remesh_botsch(
                                                                    vertices_opt.cpu().numpy().astype(np.double),
                                                                    faces_opt.cpu().numpy().astype(np.int32),
                                                                    i=1,  # Number of iterations
                                                                    h=None,  # Target edge length
                                                                    project=True
                                                                    )
                
                # Convert remeshed mesh back to PyTorch tensors
                vertices_opt = torch.tensor(vtx_pos_remeshed, dtype=torch.float32, device='cuda').contiguous()
                faces_opt = torch.tensor(pos_idx_remeshed, dtype=torch.int32, device='cuda').contiguous()

                face_normals_opt = compute_face_normals(vertices_opt, faces_opt)
                vtx_normals_opt = compute_vertex_normals(vertices_opt,  faces_opt, face_normals_opt)

                vtx_col_opt = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda').expand(vertices_opt.shape[0], -1)

                col_idx_opt = faces_opt    # Color indices match position indices
                normal_idx_opt = faces_opt    # Normal indices match position indices

        r_mvp_batch = []
        r_mvp_a_batch = []
        a_mvp_batch = []
        a_mvp_r_batch = []

        proj = util.projection(x=0.5)
        proj_opt = util.projection(x=0.5)

        camera_positions = [
        {"x_angle": np.radians(0), "y_angle": np.radians(0), "z_angle": np.radians(-180), "translation": (0.0, 0.0, 0.0)},
        {"x_angle": np.radians(0), "y_angle": np.radians(45), "z_angle": np.radians(180), "translation": (0.0, 0.0, 0.0)},
        {"x_angle": np.radians(0), "y_angle": np.radians(90), "z_angle": np.radians(180), "translation": (0.0, 0.0, 0.0)},

        {"x_angle": np.radians(0), "y_angle": np.radians(-90), "z_angle": np.radians(180), "translation": (0.0, 0.0, 0.0)},
        {"x_angle": np.radians(0), "y_angle": np.radians(-45), "z_angle": np.radians(180), "translation": (0.0, 0.0, 0.0)},
        {"x_angle": np.radians(0), "y_angle": np.radians(180), "z_angle": np.radians(180), "translation": (0.0, 0.0, 0.0)},

        {"x_angle": np.radians(-90), "y_angle": np.radians(0), "z_angle": np.radians(180), "translation": (0.0, 0.0, 0.0)},
        {"x_angle": np.radians(-45), "y_angle": np.radians(0), "z_angle": np.radians(180), "translation": (0.0, 0.0, 0.0)},
        {"x_angle": np.radians(90), "y_angle": np.radians(0), "z_angle": np.radians(180), "translation": (0.0, 0.0, 0.0)},
        ]

        for cam_params in camera_positions:
            # Generate the rotation + translation matrix for each camera
            r_rot = set_rotation_translation(
                x_angle=cam_params["x_angle"],
                y_angle=cam_params["y_angle"],
                z_angle=cam_params["z_angle"],
                translation=cam_params["translation"]
            )
            # print("Camera transformation matrix:\n", r_rot)
            r_mv = np.matmul(util.translate(0, 0.6, -4.5), r_rot)
            r_mvp = np.matmul(proj_opt, r_mv).astype(np.float32)
            r_mvp_batch.append(r_mvp)

            r_rot_a = np.matmul(util.rotate_x(0), util.rotate_y(ang))
            r_mv_a = np.matmul(util.translate(0, 0.6, -4.5), r_rot_a)
            r_mvp_a = np.matmul(proj_opt, r_mv_a).astype(np.float32)
            r_mvp_a_batch.append(r_mvp_a)


            a_mv_r = np.matmul(util.translate(0, 0.6, -4.5), r_rot)
            a_mvp_r = np.matmul(proj, a_mv_r).astype(np.float32)
            a_mvp_r_batch.append(a_mvp_r)


            a_rot = np.matmul(util.rotate_x(0), util.rotate_y(ang))
            a_mv = np.matmul(util.translate(0, 0.6, -4.5), a_rot)
            a_mvp = np.matmul(proj, a_mv).astype(np.float32)
            a_mvp_batch.append(a_mvp)

        # Convert batched MVP matrices to tensors
        r_mvp_batch = torch.tensor(np.array(r_mvp_batch, dtype=np.float32), dtype=torch.float32, device='cuda')
        r_mvp_a_batch = torch.tensor(np.array(r_mvp_a_batch, dtype=np.float32), dtype=torch.float32, device='cuda')
        a_mvp_batch = torch.tensor(np.array(a_mvp_batch, dtype=np.float32), dtype=torch.float32, device='cuda')
        a_mvp_r_batch = torch.tensor(np.array(a_mvp_r_batch, dtype=np.float32), dtype=torch.float32, device='cuda')


        # Compute edge length loss
        edge_length_loss = compute_edge_length_loss(vertices_opt, edges_tensor, reference_edge_lengths)

        # Reset reference length at each 100th iteration
        if it % 100 == 0:
            reference_edge_lengths = compute_edge_lengths(vertices_opt, edges_tensor).detach()

        ####################################################################################################

        color_opt, rast_out = render_with_lighting(glctx,
                                                r_mvp_batch,
                                                vertices_opt,
                                                faces_opt,
                                                vtx_col_opt,
                                                col_idx_opt,
                                                vtx_normals_opt,
                                                normal_idx_opt,
                                                resolution,
                                                sphere_light_positions,
                                                sphere_light_colors,
                                                ambient_color)

        color, rast_out = render_with_lighting(glctx, 
                                        a_mvp_r_batch, 
                                        vertices, 
                                        faces, 
                                        vtx_col,
                                        col_idx, 
                                        vtx_normals, 
                                        normal_idx, 
                                        resolution, 
                                        reference_light_positions, 
                                        reference_light_colors,
                                        ambient_color)
        
        pixel_loss = torch.mean((color - color_opt) ** 2)
        # pixel_loss = torch.mean(torch.sum((color - color_opt) ** 2, dim=1) ** 0.5)

        reg_loss = lambda_edge * edge_length_loss

        print(f"Iter: {it}, Pixel Loss: {pixel_loss.item()}, Regularization Loss: {reg_loss.item()}")

        # print(f"Iter: {it}, Pixel Loss: {pixel_loss.item()}")
        optimizer.zero_grad()
        pixel_loss.backward()
        reg_loss.backward()

        # torch.nn.utils.clip_grad_norm_([vertices_opt], max_norm=1.0)

        optimizer.step()
        scheduler.step()

         ####################################################################################################

        with torch.no_grad():
            # geom_loss = torch.mean(torch.sum((color - color_opt) ** 2, dim=1) ** 0.5)
            geom_loss = torch.mean((color - color_opt) ** 2)
            gl_avg.append(float(geom_loss))
            
            face_normals_opt = compute_face_normals(vertices_opt, faces_opt)
            vtx_normals_opt = compute_vertex_normals(vertices_opt, faces_opt, face_normals_opt)

        if log_interval and (it % log_interval == 0):
            gl_val = np.mean(np.asarray(gl_avg))
            gl_avg = []
            s = f"rep={1},iter={it},err={gl_val}"
            # print(s)
            if log_file:
                log_file.write(s + "\n")

        # Smooth rotation display for the top two squares
        display_image = display_interval and (it % display_interval == 0)
        save_mp4 = mp4save_interval and (it % mp4save_interval == 0)

        ang += 0.008
        if display_image or save_mp4:
            
            img_b = color[0].cpu().numpy()
            # bary = rast_out[0].detach().cpu().numpy()[::-1][..., 0:3]
            img_o = color_opt[0].detach().cpu().numpy()
            img_d, rast_out = render_with_lighting(glctx,
                                                r_mvp_a_batch,               
                                                vertices_opt,
                                                faces_opt,
                                                vtx_col_opt,
                                                col_idx_opt,
                                                vtx_normals_opt,
                                                normal_idx_opt,
                                                resolution,
                                                sphere_light_positions,   
                                                sphere_light_colors,
                                                ambient_color)
            bary2 = rast_out[0].detach().cpu().numpy()[::-1][..., 0:3]
            
            img_d = img_d[0].detach().cpu().numpy()[::-1]

            img_r, rast_out = render_with_lighting(glctx, 
                                        a_mvp_batch, 
                                        vertices, 
                                        faces, 
                                        vtx_col,
                                        col_idx, 
                                        vtx_normals, 
                                        normal_idx, 
                                        resolution, 
                                        reference_light_positions, 
                                        reference_light_colors,
                                        ambient_color)
            bary = rast_out[0].detach().cpu().numpy()[::-1][..., 0:3]
            img_r = img_r[0].cpu().numpy()[::-1]

            # result_image = make_grid(np.stack([img_o_resized, img_b_resized, img_d_resized, img_r_resized]))

            bary2 = bary2.squeeze(0)
            bary = bary.squeeze(0)

            # print(f"img_o shape: {img_o.shape}")
            # print(f"img_b shape: {img_b.shape}")
            # print(f"img_d shape: {img_d.shape}")
            # print(f"img_r shape: {img_r.shape}")
            # print(f"bary2 shape: {bary2.shape}")
            # print(f"bary shape: {bary.shape}")
            result_image = make_grid(np.stack([ img_o, img_b, img_d, img_r, bary2, bary]), ncols=2)
            # result_image = make_grid(np.stack([img_b, img_o]))

            if display_image:
                util.display_image(result_image, size=display_res, title=f'{it} / {max_iter}')
            if save_mp4 and writer is not None:
                writer.append_data(np.clip(np.rint(result_image * 255.0), 0, 255).astype(np.uint8))

    # Close file handles
    if writer is not None:
        writer.close()
    if log_file:
        log_file.close()

def main():
    parser = argparse.ArgumentParser(description='Suzanne Mesh Optimization')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--outdir', help='specify output directory', default='results')
    parser.add_argument('--max-iter', type=int, default=15)
    parser.add_argument('--display-interval', type=int, default=0)
    parser.add_argument('--mp4save-interval', type=int, default=100)
    args = parser.parse_args()

    out_dir = os.path.join(args.outdir, f"sphere_suzanne_{args.resolution}") if args.outdir else None
    if out_dir:
        print(f'Saving results under {out_dir}')
        os.makedirs(out_dir, exist_ok=True)

    optimize_cube_to_sphere(max_iter=args.max_iter, 
                           resolution=args.resolution, 
                           display_res=args.resolution, 
                           out_dir=out_dir)
    print("Done.")

if __name__ == "__main__":
    main()
