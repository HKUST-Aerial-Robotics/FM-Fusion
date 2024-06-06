#ifndef FMFUSOIN_COMMON_H
#define FMFUSOIN_COMMON_H

#include "open3d/Open3D.h"
#include "opencv2/opencv.hpp"
#include "cluster/PoseGraph.h"

namespace fmfusion
{
using namespace std;

struct InstanceConfig{
    double voxel_length = 0.02;
    double sdf_trunc = 0.04;
    open3d::camera::PinholeCameraIntrinsic intrinsic;
    int max_label_measures = 20;
    double min_voxel_weight = 2.0;
    double cluster_eps = 0.05;
    int cluster_min_points = 20;

    const std::string print_msg() const{
        std::stringstream msg;
        msg<<" - voxel_length: "<<voxel_length<<std::endl;
        msg<<" - sdf_trunc: "<<sdf_trunc<<std::endl;
        msg<<" - image shape: "<<intrinsic.width_<<", "<<intrinsic.height_<<std::endl;
        msg<<" - intrinsic fx,fy,cx,cy: "<<intrinsic.intrinsic_matrix_(0,0)<<", "<<intrinsic.intrinsic_matrix_(1,1)<<", "<<intrinsic.intrinsic_matrix_(0,2)<<", "<<intrinsic.intrinsic_matrix_(1,2)<<std::endl;
        msg<<" - max_label_measures: "<<max_label_measures<<std::endl;
        msg<<" - min_voxel_weight: "<<min_voxel_weight<<std::endl;
        msg<<" - cluster_eps: "<<cluster_eps<<std::endl;
        msg<<" - cluster_min_points: "<<cluster_min_points<<std::endl;
        return msg.str();
    }

};

struct MappingConfig{
    double depth_scale;
    double depth_max;
    int min_active_points;    

    // associations
    int min_det_masks;
    double max_box_area_ratio;
    double query_depth_vx_size; // <0 if skipped
    double search_radius;
    int dilation_size;
    double min_iou;

    // shape
    double min_voxel_weight;
    int shape_min_points=1000;

    // merge
    double merge_iou;
    double merge_inflation;
    int cleanup_period=20; // in frames

    //
    int update_period=20; // in frames

    //
    std::string save_da_dir = "";

    const std::string print_msg()const{
        std::stringstream msg;
        msg<<" - depth_scale: "<<depth_scale<<std::endl;
        msg<<" - depth_max: "<<depth_max<<std::endl;
        msg<<" - min_active_points: "<<min_active_points<<std::endl;
        msg<<" - min_det_masks: "<<min_det_masks<<std::endl;
        msg<<" - max_box_area_ratio: "<<max_box_area_ratio<<std::endl;
        msg<<" - query_depth_vx_size: "<<query_depth_vx_size<<std::endl;
        msg<<" - search_radius: "<<search_radius<<std::endl;
        msg<<" - dilation_size: "<<dilation_size<<std::endl;
        msg<<" - min_iou: "<<min_iou<<std::endl;
        msg<<" - min_voxel_weight: "<<min_voxel_weight<<std::endl;
        msg<<" - shape_min_points: "<<shape_min_points<<std::endl;
        msg<<" - merge_iou: "<<merge_iou<<std::endl;
        msg<<" - merge_inflation: "<<merge_inflation<<std::endl;
        msg<<" - cleanup_period: "<<cleanup_period<<std::endl;
        msg<<" - update_period: "<<update_period<<std::endl;
        msg<<" - save_da_dir: "<<save_da_dir<<std::endl;
        return msg.str();
    }

};

struct GraphConfig{
    float edge_radius_ratio = 2.0;
    float voxel_size = 0.02;
    bool involve_floor_edge = false;
    std::string ignore_labels = "floor. carpet. ceiling.";

    const std::string print_msg()const{
        std::stringstream msg;
        msg<<" - edge_radius_ratio: "<<edge_radius_ratio<<std::endl;
        msg<<" - voxel_size: "<<voxel_size<<std::endl;
        msg<<" - involve_floor_edge: "<<involve_floor_edge<<std::endl;
        msg<<" - ignore_labels: "<<ignore_labels<<std::endl;
        return msg.str();
    }
};

struct SgNetConfig
{
    int token_padding=8;
    int triplet_number=20; // number of triplets for each node
    float instance_match_threshold=0.1;

    const std::string print_msg()const{
        std::stringstream msg;
        msg<<" - token_padding: "<<token_padding<<std::endl;
        msg<<" - triplet_number: "<<triplet_number<<std::endl;
        msg<<" - instance_match_threshold: "<<instance_match_threshold<<std::endl;
        return msg.str();
    }
};

struct ShapeEncoderConfig
{
    int num_stages = 4;
    float voxel_size = 0.05;
    int neighbor_limits[4] = {32, 9, 9, 9};
    float init_voxel_size = 2.5 * voxel_size;
    float init_radius = 2.0 * voxel_size;
    int K_shape_samples = 1024;
    int K_match_samples = 512;
    const std::string print_msg()const{
        std::stringstream msg;
        msg<<" - num_stages: "<<num_stages<<std::endl;
        msg<<" - voxel_size: "<<voxel_size<<std::endl;
        msg<<" - neighbor_limits: "<<neighbor_limits[0]<<", "<<neighbor_limits[1]<<", "<<neighbor_limits[2]<<", "<<neighbor_limits[3]<<std::endl;
        msg<<" - init_voxel_size: "<<init_voxel_size<<std::endl;
        msg<<" - init_radius: "<<init_radius<<std::endl;
        msg<<" - K_shape_samples: "<<K_shape_samples<<std::endl;
        msg<<" - K_match_samples: "<<K_match_samples<<std::endl;
        return msg.str();
    }
};

struct LoopDetectorConfig
{
    bool fuse_features = false;
};

struct RegistrationConfig
{
    std::vector<double> noise_bound_vec = {1.0};
    const std::string print_msg() const{
        std::stringstream msg;
        msg<<" - noise_bound_vec: "<<std::fixed<<std::setprecision(2);
        for(auto &v: noise_bound_vec) msg<<v<<", ";
    
        msg<<std::endl;
        return msg.str();
    
    }
};

struct Config
{
    enum DATASET_TYPE{
        REALSENSE,
        FUSION_PORTABLE,
        SCANNET,
        MATTERPORT,
        RIO
    }dataset;
    
    InstanceConfig instance_cfg;
    MappingConfig mapping_cfg;

    // output
    // bool save_da_images;
    // std::string tmp_dir; // For debug

    //
    GraphConfig graph;
    SgNetConfig sgnet;
    ShapeEncoderConfig shape_encoder;

    //
    RegistrationConfig reg;
        
};

typedef std::shared_ptr<cv::Mat> CvMatPtr;
typedef open3d::geometry::Image O3d_Image;
typedef open3d::geometry::Geometry O3d_Geometry;
typedef std::shared_ptr<const open3d::geometry::Geometry> O3d_Geometry_Ptr;
typedef std::shared_ptr<open3d::geometry::Image> O3d_Image_Ptr;
typedef std::pair<uint32_t, uint32_t> NodePair;

}

#endif //FMFUSOIN_COMMON_H