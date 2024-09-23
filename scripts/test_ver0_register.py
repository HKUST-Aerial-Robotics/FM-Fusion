import os, glob 
import numpy as np
import open3d as o3d
import torch

from run_test_loop import read_scan_pairs
from hybrid_reg import HybridReg
from eval_loop import evaluate_fine, eval_registration_error
# from model.utils.utils import read_scan_pairs
# from model.loss.eval import Evaluator
# from omegaconf import OmegaConf

def load_matches(file_dir, pred_node_pairs, src_points, ref_points, gt_mask):
    with open(file_dir,'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0]=='#':
                continue
            items = line.split(' ')
            pred_node_pairs.append([int(items[0]),int(items[1])])
            src_points.append([float(items[3]),float(items[4]),float(items[5])])
            ref_points.append([float(items[6]),float(items[7]),float(items[8])])
            gt_mask.append(int(items[2]))
        f.close()
        pred_node_pairs = np.array(pred_node_pairs)
        src_points = np.array(src_points).astype(np.float32)
        ref_points = np.array(ref_points).astype(np.float32)
        gt_mask = np.array(gt_mask)
        
        return pred_node_pairs, src_points, ref_points, gt_mask

if __name__=='__main__':
    ############ Args ############
    dataroot = '/data2/RioGraph'
    split_file = 'val'
    output_folder = os.path.join(dataroot, 'output', 'ours')
    ##############################

    RUN_REG = True
    
    # 
    scene_pairs = read_scan_pairs(os.path.join(dataroot, 'splits', f'{split_file}.txt'))
    recall_list = []
    
    for pair in scene_pairs:
        print('------------- {}-{} -------------'.format(pair[0], pair[1]))
        pair_folder = os.path.join(output_folder, f'{pair[0]}-{pair[1]}')
        src_pcd = o3d.io.read_point_cloud(os.path.join(pair_folder, 'src_instances.ply'))
        ref_pcd = o3d.io.read_point_cloud(os.path.join(pair_folder, 'ref_instances.ply'))
        
        corr_src_pcd = o3d.io.read_point_cloud(os.path.join(pair_folder, 'corr_src.ply'))
        corr_ref_pcd = o3d.io.read_point_cloud(os.path.join(pair_folder, 'corr_ref.ply'))
        corr_inst_scores = np.loadtxt(os.path.join(pair_folder, 'point_matches.txt'))
        corr_instances = corr_inst_scores[:,0].astype(np.int32)
        corr_scores = corr_inst_scores[:,1].astype(np.float32)
        assert(corr_inst_scores.shape[0]==len(corr_src_pcd.points))
        _, src_nodes, ref_nodes, _ = load_matches(os.path.join(pair_folder, 'node_matches.txt'), [], [], [], [])
        # print('Load {} nodes pairs'.format(len(src_nodes)))
        M = src_nodes.shape[0]
            
        gt_transform = np.loadtxt(os.path.join(pair_folder, 'gt_pose.txt'))
        
        if False: # dont use it. Just for transmit gt file
            np.savetxt(os.path.join(pair_folder, 'gt_pose.txt'), gt_transform) 
            continue   

        inlier_ratio , _ = evaluate_fine(np.asarray(corr_src_pcd.points), 
                                         np.asarray(corr_ref_pcd.points), 
                                         gt_transform,
                                         acceptance_radius=0.1)
    
        if RUN_REG:
            hybrid_reg = HybridReg(
                src_pcd,
                ref_pcd,
                refine=None, #"vgicp",
                use_pagor=False,
                only_yaw=True,
                max_ins=128,
                max_pts=32,
                ins_wise=False,
            )
            
            hybrid_reg.set_model_pred({'corres_src_points': np.asarray(corr_src_pcd.points).astype(np.float32),
                                       'corres_ref_points': np.asarray(corr_ref_pcd.points).astype(np.float32),
                                       'corres_scores': corr_scores,
                                       'corres_instances': corr_instances,
                                       'pred_scores':0.2*np.ones(M),
                                       'corres_src_centroids': src_nodes,
                                       'corres_ref_centroids': ref_nodes})
            estimated_transform = hybrid_reg.solve()
            np.savetxt(os.path.join(pair_folder, 'pred_pose_ver0.txt'), 
                       estimated_transform)
            rmse = eval_registration_error(src_pcd, gt_transform, estimated_transform)
            if rmse<0.2:
                recall = 1.0
            else:
                recall = 0.0
        else:
            rmse = 10.0
            recall = 0.0
    
        print('Inlier ratio: {:.1f}, rmse: {:.3f}m '.format(inlier_ratio.item()*100,
                                                            rmse))
        recall_list.append(recall)
        # break
    
    print('Finished {} scene pairs'.format(len(scene_pairs)))
    if (len(recall_list)>0):
        recall_list = np.array(recall_list)
        print('Mean recall: {:.3f}({}/{})'.format(np.mean(recall_list),
                                                  recall_list.sum(),
                                                  recall_list.shape[0]))