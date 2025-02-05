import argparse
import os
import pathlib
import numpy as np
import torch
import imageio
import util
import nvdiffrast.torch as dr
import torch.nn.functional as F

# Transform vertex positions to clip space
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def apply_lighting(normals, interpolated_light_dirs, light_colors, ambient_color):
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

    return shading

def load_obj(filepath):
    vertices = []
    faces = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                face = []
                parts = line.strip().split()
                for v in parts[1:]:
                    v_parts = v.split('/')
                    face.append(int(v_parts[0]) - 1)  # Vertex index
                # Handle quads by splitting into triangles
                if len(face) == 3:
                    faces.append(face)
                elif len(face) == 4:
                    faces.append([face[0], face[1], face[2]])
                    faces.append([face[0], face[2], face[3]])
    
    vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    faces = torch.tensor(faces, dtype=torch.int32, device='cuda')
    return vertices, faces

# Function to compute face normals
def compute_face_normals(vtx_pos, tri):
    idx0 = tri[:, 0]
    idx1 = tri[:, 1]
    idx2 = tri[:, 2]
    v0 = vtx_pos[idx0]
    v1 = vtx_pos[idx1]
    v2 = vtx_pos[idx2]
    # Compute two edges of the triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    # Compute the normal (cross product)
    face_normals = torch.cross(edge1, edge2, dim=1)
    # Normalize the normals
    face_normals = F.normalize(face_normals, dim=1)
    return face_normals

# Function to compute per-vertex normals by averaging adjacent face normals
def compute_vertex_normals(vtx_pos, tri, face_normals):
    num_vertices = vtx_pos.shape[0]
    vertex_normals = torch.zeros(num_vertices, 3, device='cuda')
    count = torch.zeros(num_vertices, 1, device='cuda')
    for i in range(tri.shape[0]):
        face = tri[i]
        for idx in face:
            vertex_normals[idx] += face_normals[i]
            count[idx] += 1
    # Avoid division by zero
    count[count == 0] = 1
    vertex_normals /= count
    # Normalize the vertex normals
    vertex_normals = F.normalize(vertex_normals, dim=1)
    return vertex_normals

# Rendering with lighting using multiple lights and vertex normals
def render_with_lighting(glctx, mtx, pos, pos_idx, vtx_col, col_idx, vtx_normals, normal_idx, resolution, light_positions, light_colors, ambient_color):
    pos_clip = transform_pos(mtx, pos).contiguous()
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx.contiguous(), resolution=[resolution, resolution])
    color, _ = dr.interpolate(vtx_col[None, ...].contiguous(), rast_out.contiguous(), col_idx.contiguous())
    normal, _ = dr.interpolate(vtx_normals[None, ...].contiguous(), rast_out.contiguous(), normal_idx.contiguous())
    color = dr.antialias(color.contiguous(), rast_out.contiguous(), pos_clip, pos_idx.contiguous())

    # Convert light positions to tensors and expand to match vertex positions
    light_positions = [torch.tensor(light_position, dtype=torch.float32).cuda() for light_position in light_positions]
    light_colors = [torch.tensor(light_color, dtype=torch.float32).cuda() for light_color in light_colors]

    # Interpolate light direction across the rasterized space for each light
    interpolated_light_dirs = []
    for light_position in light_positions:
        light_dir = F.normalize(light_position - pos, dim=-1)
        light_dir_interpolated, _ = dr.interpolate(light_dir[None, ...].contiguous(), rast_out.contiguous(), pos_idx.contiguous())
        interpolated_light_dirs.append(light_dir_interpolated)

    # Apply lighting
    ambient_color = torch.tensor(ambient_color, dtype=torch.float32).cuda()

    shaded_color = apply_lighting(normal, interpolated_light_dirs, light_colors, ambient_color)

    # Multiply the vertex color with the lighting result
    shaded_color = color * shaded_color[None, ...].contiguous()

    return shaded_color, rast_out

def rasterize_mesh_with_lighting(resolution=512, display_res=512, out_dir=None):
    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'
    obj_file = f'{datadir}/suzanne.obj'

    # Load vertices, normals, faces, and normal indices from the OBJ file
    vertices, faces = load_obj(obj_file)

    print("Mesh has %d triangles and %d vertices." % (faces.shape[0], vertices.shape[0]))

    pos_idx = faces
    vtx_pos = vertices

    # Compute normals - faces are not deformed so one compute is adequate
    face_normals = compute_face_normals(vtx_pos, pos_idx)
    vtx_normals = compute_vertex_normals(vtx_pos, pos_idx, face_normals)

    # Colors for each vertex (assuming a constant color)
    vtx_col = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda').expand(vertices.shape[0], -1)

    col_idx = pos_idx       # Color indices match position indices
    normal_idx = pos_idx    # Normal indices match position indices

    glctx = dr.RasterizeCudaContext()

    # Light settings for three different lights
    light_positions = [
        [-5.0, 5.0, -5.0],
        [5.0, -5.0, 5.0]
    ]
    light_colors = [
        [1.0, 0.0, 0.0],  
        [0.0, 1.0, 0.0],  
    ]
    
    ambient_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device='cuda')

    # Rotation parameters
    num_frames = 1000  # Number of frames to render
    rotation_speed = np.pi / 200  # Rotation angle per frame

    for frame in range(num_frames):
        angle = frame * rotation_speed
        rotation_matrix = np.matmul(util.rotate_x(0.4), util.rotate_y(angle))  # Rotate around Y-axis

        # Modelview matrix: apply rotation and translation
        mv = np.matmul(util.translate(0, 0.6, -4.5), rotation_matrix)
        proj = util.projection(x=0.5)
        mvp = np.matmul(proj, mv).astype(np.float32)

        # Render the mesh with the current rotation and lighting from 3 different lights
        color, rast_out = render_with_lighting(
            glctx,
            mvp,              
            vtx_pos,
            pos_idx,
            vtx_col,
            col_idx,
            vtx_normals,
            normal_idx,
            resolution,
            light_positions,  
            light_colors,
            ambient_color
        )
        img = color[0].detach().cpu().numpy()[::-1][..., 0:3]
        img = np.squeeze(img)  # Remove any batch dimension (shape should be [512, 512, 3])

        # Ensure the image is in the correct shape for saving (H, W, 3)
        if img.shape[-1] == 3: 
            result_image = np.repeat(np.repeat(img, display_res // img.shape[0], axis=0), display_res // img.shape[1], axis=1)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            imageio.imwrite(f'{out_dir}/frame_{frame:03d}.png', np.clip(result_image * 255.0, 0, 255).astype(np.uint8))

        util.display_image(result_image, size=display_res, title=f'Frame {frame}/{num_frames}')


def main():
    parser = argparse.ArgumentParser(description='rasterization')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--outdir', help='specify output directory', default='results')
    args = parser.parse_args()

    if args.outdir:
        out_dir = f'{args.outdir}/suzanne_{args.resolution}'
        print(f'Saving results under {out_dir}')
    else:
        out_dir = None
        print('No output directory specified, not saving image')

    rasterize_mesh_with_lighting(resolution=args.resolution, out_dir=out_dir)
    print("Done.")


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

