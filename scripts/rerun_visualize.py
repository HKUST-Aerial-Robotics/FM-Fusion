import os 
import numpy as np
import cv2
import open3d as o3d
import rerun as rr
import argparse, json
from scipy.spatial.transform import Rotation as R

from os.path import join as osp

from utils import load_processed_scene_graph, load_raw_scene_graph

def render_point_cloud(entity_name:str,
                        cloud:o3d.geometry.PointCloud, 
                        radius=0.1,
                       color=None):
    """
    Render a point cloud with a specific color and point size.
    """
    if color is not None:
        viz_colors = color
    else:   
        viz_colors = np.asarray(cloud.colors)

    rr.log(entity_name,
           rr.Points3D(
               np.asarray(cloud.points),
               colors=viz_colors,
               radii=radius,
           )
           )

def render_node_centers(entity_name:str,
                        nodes:dict,
                        radius=0.01,
                        color=[255,255,255]):
    """
    Render the centers of nodes in the scene graph.
    """
    
    centers = []
    semantic_labels = []
    for node in nodes.values():
        if isinstance(node, o3d.geometry.OrientedBoundingBox):
            centers.append(node.center)
        else:
            centers.append(node.cloud.get_center())
        semantic_labels.append(node.label)
    centers = np.array(centers)
    if color is None:
        raise NotImplementedError('todo: color by node pcd')
    else:
        viz_colors = color
    rr.log(entity_name,
           rr.Points3D(
               centers,
               colors=viz_colors,
               radii=radius,
            #    labels=semantic_labels,
               show_labels=False
           )
           )

def render_node_bboxes(entity_name:str,
                       nodes:dict,
                       show_labels:bool=True,
                       radius=0.01):
    
    for idx, node in nodes.items():
        rot = node.box.R
        quad = R.from_matrix(rot.astype(np.float32)).as_quat()
        color = np.asarray(node.cloud.colors) *255
        
        rr.log('{}/{}'.format(entity_name,idx),
               rr.Boxes3D(half_sizes=0.5*node.box.extent,
                          centers=node.box.center,
                          quaternions=rr.Quaternion(xyzw=quad),
                          radii=radius,
                          colors=color.astype(np.uint8),
                          labels=node.label,
                          show_labels=show_labels)
               )

def render_camera_pose(scene_dir:str,
                       frame_name:str,
                       entity_name:str='world/camera',
                       dist:float=1.0):
    w, h = 640, 480
    fx, fy = 580, 580    
    pose_dir = osp(scene_dir,'pose',frame_name+'.txt')
    rgb_dir  = osp(scene_dir,'color',frame_name+'.jpg')
    pose = np.loadtxt(pose_dir)
    rgb = cv2.imread(rgb_dir)
        
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    INSTRINSIC = np.array([[fx,0,w/2],
                            [0,fy,h/2],
                            [0,0,1]])    
    print('Load camera pose from ',pose_dir)
    
    rr.log(entity_name,
           rr.Transform3D(translation=pose[:3,3],
                          mat3x3=pose[:3,:3])
           )
    rr.log(entity_name,
           rr.Pinhole(resolution=[w,h],
                      image_from_camera=INSTRINSIC,
                      camera_xyz=rr.ViewCoordinates.RDF,
                      image_plane_distance=dist),
           )
    rr.log(entity_name,
           rr.EncodedImage(path=rgb_dir))
    
    rr.log('image',rr.Image(rgb))
    
def project2_image(points:np.ndarray,
                   colors:np.ndarray,
                   scene_dir:str,
                   frame_name:str,
                   entity_name:str="world/render_img"):
    # 
    w, h = 640, 480
    fx, fy = 577.870605, 577.870605
    INSTRINSIC = np.array([[fx,0,w/2],
                            [0,fy,h/2],
                            [0,0,1]])
    pose_dir = osp(scene_dir,'pose',frame_name+'.txt')
    pose = np.loadtxt(pose_dir) # T_w_c
    print('Load camera pose from ',pose_dir)
    
    # 
    proj_image = np.zeros((h,w,3),dtype=np.uint8)
    proj_image.fill(255)
    
    # Transform points to camera frame
    N = points.shape[0]
    points = np.hstack((points,np.ones((points.shape[0],1)))) # N x 4
    points = np.linalg.inv(pose) @ points.T
    points = points.T[:,:3] # N x 3

    # Project points to image plane
    points = points[:,:3] / points[:,2:] # N x 3
    points = points @ INSTRINSIC.T # N x 3
    points = points[:,:2] / points[:,2:] # N x 2
    points = points.astype(np.int32) # [u, v]
    points[:,0] = np.clip(points[:,0],0,w-1)
    points[:,1] = np.clip(points[:,1],0,h-1)

    # Draw points
    RADIUS = 5
    print('Color range:',colors.min(),colors.max())
    
    for i in range(points.shape[0]):
        color = (colors[i] * 255).astype(np.uint8).tolist()
        assert len(color)==3, 'color should be 3 channels'
        cv2.circle(proj_image,
                   (points[i,0],points[i,1]),
                   RADIUS,
                #    [0,0,255],
                    [color[2],color[1],color[0]],
                   -1)
    
    # Render points on image
    rr.log(entity_name, rr.Image(proj_image))
    
    #
    cv2.imwrite(osp(scene_dir,'render',frame_name+'.jpg'),
                proj_image)
    print('Save image to ',osp(scene_dir,'render',frame_name+'.jpg'))
    
def render_semantic_scene_graph(scene_name:str,
                                scene_graph:dict,
                                voxel_size:float=0.05,
                                origin:np.ndarray=np.eye(4),
                                box:bool=False
                                ):
    render_point_cloud(scene_name+'/global_cloud',
                       scene_graph['global_cloud'],
                       voxel_size)
    render_node_centers(scene_name+'/centroids',
                        scene_graph['nodes'])
    
    if box:
        render_node_bboxes(scene_name+'/nodes',
                            scene_graph['nodes'],
                            show_labels=True)
        
    quad = R.from_matrix(origin[:3,:3]).as_quat()
    rr.log(scene_name+'/local_origin',
            rr.Transform3D(translation=origin[:3,3],
                            quaternion=quad)
            )
        
def render_correspondences(entity_name:str,
                           src_points:np.ndarray,
                           ref_points:np.ndarray,
                           transform:np.ndarray=None,
                           gt_mask:np.ndarray=None,
                           radius=0.01):
    
    N = src_points.shape[0]
    assert N==ref_points.shape[0], 'src and ref points should have the same number of points'
    line_points = []
    line_colors = []

    for i in range(N):
        src = src_points[i]
        ref = ref_points[i]
        if transform is not None:
            src = transform[:3,:3] @ src + transform[:3,3]
            
        if gt_mask[i]:
            line_colors.append([0,255,0])
        else:
            line_colors.append([255,0,0])
    
        line_points.append([src,ref])
        
    
    line_points = np.concatenate(line_points,axis=0)
    line_points = line_points.reshape(-1,2,3)
    line_colors = np.array(line_colors)
    rr.log(entity_name,
           rr.LineStrips3D(line_points,
                           radii=radius,
                           colors=line_colors)
           )

def render_registration(entity_name:str,
                        src_cloud:o3d.geometry.PointCloud,
                        ref_cloud:o3d.geometry.PointCloud,
                        transform:np.ndarray):
    
    src_cloud.transform(transform)
    src_points = np.asarray(src_cloud.points)
    ref_points = np.asarray(ref_cloud.points)
    src_color = [0,180,180]
    ref_color = [180,180,0]
    rr.log(entity_name+'/src',
           rr.Points3D(src_points,
                       colors=src_color,
                       radii=0.01)
           )
    rr.log(entity_name+'/ref',
           rr.Points3D(ref_points,
                       colors=ref_color,
                       radii=0.01)
           )
           

def get_parser_args():
    def float_list(string):
        return [float(x) for x in string.split(',')]
    
    parser = argparse.ArgumentParser(description='Visualize scene graph')
    parser.add_argument('--src_scene_dir', type=str, required=True,
                        help='source scene name')
    parser.add_argument('--viz_mode', type=int, required=True,
                        help='0: no viz, 1: local viz, 2: remote viz, 3: save rrd')
    parser.add_argument('--seq_dir', type=str, default=None,
                        help='only used to load selected frame')
    parser.add_argument('--frame_name', type=str, default=None,)
    parser.add_argument('--remote_rerun_add', type=str, help='IP:PORT')
    parser.add_argument('--voxel_size', type=float, default=0.02,
                        help='voxel size for downsampling')    
    
    return parser.parse_args()

if __name__=='__main__':
    print('*'*60)
    print('This script reads the data association and registration results.')
    print('*'*60)
    
    ############ Args ############
    args = get_parser_args()
    SPLIT = 'val'
    print('Visualize {}'.format(args.src_scene_dir))
    ##############################

    # Load scene graphs
    src_sg = load_raw_scene_graph(args.src_scene_dir)

    # Stream to rerun
    rr.init("SGReg")
    render_semantic_scene_graph('src',
                                src_sg,
                                args.voxel_size,
                                np.eye(4),
                                True)
    if args.frame_name is not None:
        assert args.seq_dir is not None, \
            'require a sequence dir to load selected frame'
        render_camera_pose(args.seq_dir,
                           args.frame_name)
        
        project2_image(np.asarray(src_sg['global_cloud'].points),
                       np.asarray(src_sg['global_cloud'].colors),
                       args.seq_dir,
                       args.frame_name)
    
    # Render on rerun
    if args.viz_mode==1:
        rr.spawn()
    elif args.viz_mode==2:
        assert args.remote_rerun_add is not None, \
            'require a remote address for rendering, (eg. 143.89.38.169:9876)'
        print('--- Render rerun at a remote machine ',args.remote_rerun_add, '---')
        rr.connect_tcp(args.remote_rerun_add)
    elif args.viz_mode==3:
        rr.save(osp(args.src_scene_dir,'result.rrd'))
        print('Save rerun data to ',osp(args.src_scene_dir,'result.rrd'))
    else:
        print('No visualization')