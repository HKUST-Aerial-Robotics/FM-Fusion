import os
import torch
import json
import pandas as pd
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from open3d.geometry import OrientedBoundingBox as OBB
from os.path import join as osp

class Instance:
    def __init__(self,idx:int,
                 cloud:o3d.geometry.PointCloud,
                 label:str,
                 score:float):
        self.idx = idx
        self.label = label
        self.score = score
        self.cloud = cloud
        self.cloud_dir = None
    def load_box(self,box:o3d.geometry.OrientedBoundingBox):
        self.box = box

def load_raw_scene_graph(folder_dir:str,
                     voxel_size:float=0.02,
                     ignore_types:list=['ceiling']):
    ''' graph: {'nodes':{idx:Instance},'edges':{idx:idx}}
    '''
    # load scene graph
    nodes = {}
    boxes = {}
    invalid_nodes = []
    xyzi = []
    global_cloud = o3d.geometry.PointCloud()
    # IGNORE_TYPES = ['floor','carpet','wall']
    # IGNORE_TYPES = ['ceiling']
    
    # load instance boxes
    with open(os.path.join(folder_dir,'instance_box.txt')) as f:
        count=0
        for line in f.readlines():
            line = line.strip()
            if'#' in line:continue
            parts = line.split(';')
            idx = int(parts[0])
            center = np.array([float(x) for x in parts[1].split(',')])
            rotation = np.array([float(x) for x in parts[2].split(',')])
            extent = np.array([float(x) for x in parts[3].split(',')])
            o3d_box = o3d.geometry.OrientedBoundingBox(center,rotation.reshape(3,3),extent)
            o3d_box.color = (0,0,0)
            # if'nan' in line:invalid_nodes.append(idx)
            if 'nan' not in line:
                boxes[idx] = o3d_box
                # nodes[idx].load_box(o3d_box)
                count+=1
        f.close()
        print('load {} boxes'.format(count))    
        
    # load instance info
    with open(os.path.join(folder_dir,'instance_info.txt')) as f:
        for line in f.readlines():
            line = line.strip()
            if'#' in line:continue
            parts = line.split(';')
            idx = int(parts[0])
            if idx not in boxes: continue
            label_score_vec = parts[1].split('(')
            label = label_score_vec[0]
            score = float(label_score_vec[1].split(')')[0])
            if label in ignore_types: continue
            # print('load {}:{}, {}'.format(idx,label,score))
            
            cloud = o3d.io.read_point_cloud(os.path.join(folder_dir,'{}.ply'.format(parts[0])))
            cloud = cloud.voxel_down_sample(voxel_size)
            xyz = np.asarray(cloud.points)
            # if xyz.shape[0]<50: continue
            xyzi.append(np.concatenate([xyz,idx*np.ones((len(xyz),1))],
                                       axis=1))
            global_cloud = global_cloud + cloud
            nodes[idx] = Instance(idx,cloud,label,score)
            nodes[idx].cloud_dir = '{}.ply'.format(parts[0])
            nodes[idx].load_box(boxes[idx])

        f.close()
        print('Load {} instances '.format(len(nodes)))
    if len(xyzi)>0:
        xyzi = np.concatenate(xyzi,axis=0)
        
    return {'nodes':nodes,
            'edges':[],
            'global_cloud':global_cloud, 
            'xyzi':xyzi}

def load_processed_scene_graph(scan_dir:str):
    
    instance_nodes = {} # {idx:node_info}
    
    # Nodes
    nodes_data = pd.read_csv(os.path.join(scan_dir,'nodes.csv'))
    max_node_id = 0

    for idx, label, score, center, quat, extent, _ in zip(nodes_data['node_id'],
                                                                    nodes_data['label'],
                                                                    nodes_data['score'],
                                                                    nodes_data['center'],
                                                                    nodes_data['quaternion'],
                                                                    nodes_data['extent'],
                                                                    nodes_data['cloud_dir']):
        centroid = np.fromstring(center, dtype=float, sep=',')
        quaternion = np.fromstring(quat, dtype=float, sep=',') # (x,y,z,w)
        rot = R.from_quat(quaternion)
        extent = np.fromstring(extent, dtype=float, sep=',')
                
        if np.isnan(extent).any() or np.isnan(quaternion).any() or np.isnan(centroid).any() or np.isnan(idx):
            continue
        if '_' in label: label = label.replace('_',' ')
        pcd_dir = osp(scan_dir, str(idx).zfill(4)+'.ply')
        assert os.path.exists(pcd_dir), 'File not found: {}'.format(pcd_dir)
        
        instance_nodes[idx] = Instance(idx, 
                                       o3d.io.read_point_cloud(pcd_dir),
                                       label, 
                                       score)
        instance_nodes[idx].load_box(OBB(centroid, 
                                         rot.as_matrix(),
                                         extent))
        
        if idx>max_node_id:
            max_node_id = idx
    print('Load {} instances'.format(len(instance_nodes)))

    # Instance Point Cloud
    xyzi = torch.load(os.path.join(scan_dir,'xyzi.pth')).numpy()
    instances = xyzi[:,-1].astype(np.int32)
    xyz = xyzi[:,:3].astype(np.float32)
    assert max_node_id == instances.max(), 'Instance ID mismatch'
    assert np.unique(instances).shape[0] == len(instance_nodes), 'Instance ID mismatch'
    
    # Global Point Cloud
    # colors = np.zeros_like(xyz)
    # for idx, instance in instance_nodes.items():  
    #     inst_mask = instances== idx
    #     assert inst_mask.sum()>0
    #     inst_color = 255*np.random.rand(3)
    #     colors[inst_mask] = np.floor(inst_color).astype(np.int32)
    #     instance.cloud = o3d.geometry.PointCloud(
    #                     o3d.utility.Vector3dVector(xyz[inst_mask]))
    
    # global_pcd = o3d.geometry.PointCloud(
    #             o3d.utility.Vector3dVector(xyz))
    # global_pcd.colors = o3d.utility.Vector3dVector(colors)
    # global_pcd = o3d.io.read_point_cloud(os.path.join(scan_dir,'instance_map.ply'))
    
    global_pcd = o3d.geometry.PointCloud()
    for idx, instance in instance_nodes.items():
        global_pcd += instance.cloud

    return {'nodes':instance_nodes,
            'edges':[],
            'global_cloud':global_pcd}


def transform_scene_graph(scene_graph:dict,
                          transformation:np.ndarray):
    
    scene_graph['global_cloud'].transform(transformation)
    for idx, instance in scene_graph['nodes'].items():
        tmp_center = instance.box.center
        instance.cloud.transform(transformation)
        # todo: open3d rotate the bbox falsely 
        # instance.box = o3d.geometry.OrientedBoundingBox.create_from_points(
        #     o3d.utility.Vector3dVector(np.asarray(instance.cloud.points)))
        instance.box.rotate(R=transformation[:3,:3])
        instance.box.translate(transformation[:3,3])
        



def read_data_association(da_file:str):
    assert os.path.exists(da_file), 'File not found: {}'.format(da_file)
    with open(da_file, 'r') as f:
        depth_frames = []
        rgb_frames = []
        lines = f.readlines()

        for line in lines:
            depth_frame, rgb_frame = line.strip().split(' ')
            depth_frames.append(depth_frame)
            rgb_frames.append(rgb_frame)
        
        return depth_frames, rgb_frames

def read_scan_pairs(dir:str):
    with open(dir) as f:
        lines = f.readlines()
        pairs = []
        for line in lines:
            src_scan, ref_scan = line.strip().split(' ')
            pairs.append((src_scan, ref_scan))
        f.close()
        return pairs

def read_ram_tags(json_file_dir:str):
    assert os.path.exists(json_file_dir), 'File not found: {}'.format(json_file_dir)
    with open(json_file_dir, 'r') as f:
        data = json.load(f)
        if 'raw_tags' in data:
            tags_str = data['raw_tags']
            # return tags_str
            elements = tags_str.split('.')
            tags = []
            for element in elements:
                if element == '':
                    continue
                tags.append(element.strip())
            return tags
        else:
            return []