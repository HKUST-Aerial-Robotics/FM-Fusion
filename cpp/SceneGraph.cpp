#include "SceneGraph.h"

namespace fmfusion
{

SceneGraph::SceneGraph(const Config &config): config_(config)
{
    open3d::utility::LogInfo("Initialize SceneGraph server");
    instance_config.voxel_length = config_.voxel_length;
    instance_config.sdf_trunc = config_.sdf_trunc;
    instance_config.intrinsic = config_.intrinsic;
    instance_config.min_voxel_weight = config_.min_voxel_weight;
}

void SceneGraph::integrate(const int &frame_id,
    const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose,
    std::vector<DetectionPtr> &detections)
{
    open3d::utility::LogInfo("## Integrate SceneGraph");
    int n_det = detections.size();

    if(n_det>1){
        o3d_utility::Timer timer_query, timer_da, timer_integrate;
        timer_query.Start();
        // O3d_Image_Ptr depth_continuity_mask = rgbd_image->depth_.CreateDepthBoundaryMask();
        // O3d_Image_Ptr filtered_depth = extract_masked_o3d_image(rgbd_image->depth_,*depth_continuity_mask);

        auto depth_cloud = O3d_Cloud::CreateFromDepthImage(rgbd_image->depth_,config_.intrinsic,pose.inverse(),1.0,config_.depth_max);
        auto active_instances = search_active_instances(depth_cloud,pose);
        timer_query.Stop();

        timer_da.Start();
        auto matches = data_association(detections, active_instances);
        timer_da.Stop();
        // std::cout<<"matches: "<<matches.transpose()<<std::endl;

        timer_integrate.Start();
        for (int k_=0;k_<n_det;k_++){
            auto masked_rgbd = std::make_shared<open3d::geometry::RGBDImage>();
            bool valid_detection_depth = utility::create_masked_rgbd(
                rgbd_image->color_,rgbd_image->depth_,detections[k_]->instances_idxs_,config_.min_det_masks,masked_rgbd);

            if(!valid_detection_depth){
                matches(k_) = -1;
                continue;
            }

            if (matches(k_)>0){
                auto matched_instance = instance_map[matches(k_)];
                matched_instance->integrate(masked_rgbd,pose.inverse());
                matched_instance->update_label(detections[k_]);
            }
            else{
                create_new_instance(detections[k_], frame_id,masked_rgbd,pose);
            }
        }
        timer_integrate.Stop();

        // Visualize associations
        auto rgb_cv = std::make_shared<cv::Mat>(rgbd_image->color_.height_,rgbd_image->color_.width_,CV_8UC3);
        memcpy(rgb_cv->data,rgbd_image->color_.data_.data(),rgbd_image->color_.data_.size()*sizeof(uint8_t));
        cv::cvtColor(*rgb_cv,*rgb_cv,cv::COLOR_RGB2BGR);

        std::stringstream ss, msg;
        ss<<config_.tmp_dir<<"/tmp/frame-"<<std::fixed<<std::setw(6)<<std::setfill('0')<<frame_id<<".png";
        std::unordered_map<InstanceId,CvMatPtr> active_instance_masks;
        std::unordered_map<InstanceId,Eigen::Vector3d> active_instance_colors;
        msg<<"active instances: ";
        for(auto &instance_j:instance_map){
            if(instance_j.second->observed_image_mask){
                active_instance_masks.emplace(instance_j.first,instance_j.second->observed_image_mask);
                active_instance_colors.emplace(instance_j.first,instance_j.second->color_);
                msg<<instance_j.second->get_predicted_class().first<<", ";
            }
        }
        // std::cout<<msg.str()<<"\n";

        if(config_.save_da_images && !config_.tmp_dir.empty()){
            auto render_img = utility::RenderDetections(rgb_cv,detections,active_instance_masks,matches, active_instance_colors);
            cv::imwrite(ss.str(),*render_img);
        }
        //
        update_active_instances(active_instances);
        
        // Manage recent observed instances
        for(InstanceId j_:active_instances){
            recent_instances.emplace(j_);
        }

        o3d_utility::LogInfo("## {}/{} instances",active_instances.size(),instance_map.size());
        o3d_utility::LogInfo("Time record (ms): query {:f}, da {:f}, integrate {:f}",
            timer_query.GetDurationInMillisecond(),timer_da.GetDurationInMillisecond(),timer_integrate.GetDurationInMillisecond());
    }

}

std::vector<InstanceId> SceneGraph::search_active_instances(
    const O3d_Cloud_Ptr &depth_cloud, const Eigen::Matrix4d &pose)
{
    std::vector<InstanceId> active_instances;
    const size_t MIN_UNITS= 1;

    for(auto &instance_j:instance_map){
        // Eigen::Vector3d centroid_cam_j = pose.block<3,3>(0,0)*instance_j.second->centroid+pose.block<3,1>(0,3);
        // if(centroid_cam_j(2)<-2.0) continue;

        auto observed_cloud = std::make_shared<O3d_Cloud>();
        instance_j.second->get_volume()->query_observed_points(depth_cloud,observed_cloud);
        if(observed_cloud->points_.size()>config_.min_instance_points){
            std::shared_ptr<cv::Mat> instance_img_mask = utility::PrjectionCloudToDepth(
                *observed_cloud,pose.inverse(),instance_config.intrinsic,config_.dilation_size);
            instance_j.second->observed_image_mask = instance_img_mask;
            active_instances.emplace_back(instance_j.first);
        }
    }

    return active_instances;
}

void SceneGraph::update_active_instances(const std::vector<InstanceId> &active_instances)
{
    for(InstanceId j_:active_instances){
        auto instance_j = instance_map[j_];
        instance_j->observed_image_mask.reset();
        instance_j->fast_update_centroid();
    }
}

Eigen::VectorXi SceneGraph::data_association(const std::vector<DetectionPtr> &detections,
    const std::vector<InstanceId> &active_instances)
{
    int K = detections.size();
    int M = active_instances.size();

    Eigen::VectorXi matches = Eigen::VectorXi::Zero(K);
    if (M<1) return matches;

    Eigen::MatrixXd iou = Eigen::MatrixXd::Zero(K,M);
    Eigen::MatrixXi assignment = Eigen::MatrixXi::Zero(K,M);
    Eigen::MatrixXi assignment_colwise = Eigen::MatrixXi::Zero(K,M);
    Eigen::MatrixXi assignment_rowise = Eigen::MatrixXi::Zero(K,M);

    for (int k_=0;k_<K;k_++){
        const auto &zk = detections[k_];
        double zk_area = double(cv::countNonZero(zk->instances_idxs_));
        for (int m_=0;m_<M;m_++){
            auto instance_m = instance_map[active_instances[m_]];
            cv::Mat overlap = instance_m->observed_image_mask->mul(zk->instances_idxs_);
            double overlap_area = double(cv::countNonZero(overlap));
            double instance_area = double(cv::countNonZero(*instance_m->observed_image_mask));
            iou(k_,m_) = overlap_area/(zk_area+instance_area-overlap_area);
            // iou(k_,m_) = double(cv::countNonZero(overlap))/double(cv::countNonZero(*instance_m->observed_image_mask));   // overlap/r_m
        }
    } 

    // Find the maximum match for each colume
    for (int m_=0;m_<M;m_++){
        int max_row;
        double max_iou = iou.col(m_).maxCoeff(&max_row);
        if (max_iou>config_.min_iou)assignment_colwise(max_row,m_) = 1;
    }

    // Find the maximum match for each row
    for (int k_=0;k_<K;k_++){
        int max_col;
        double max_iou = iou.row(k_).maxCoeff(&max_col);
        if (max_iou>config_.min_iou)assignment_rowise(k_,max_col) = 1;
    }

    assignment = assignment_colwise + assignment_rowise;

    // export matches
    int count= 0;
    for (int k_=0;k_<K;k_++){
        for (int m_=0;m_<M;m_++){
            if (assignment(k_,m_)==2){
                matches(k_) = active_instances[m_];
                count ++;
                break;
            }
        }
    }

    // std::cout<<iou<<std::endl;   
    // std::cout<<assignment<<std::endl;
    o3d_utility::LogInfo("{}/({},{}) associations out of detections and active instances.",count,K,M);

    return matches;
}

bool SceneGraph::create_new_instance(const DetectionPtr &detection, const unsigned int &frame_id,
    const std::shared_ptr<open3d::geometry::RGBDImage> &rgbd_image, const Eigen::Matrix4d &pose)
{
    // open3d::utility::LogInfo("Create new instance");
    auto instance = std::make_shared<Instance>(instance_map.size()+1,frame_id,instance_config);
    instance->integrate(rgbd_image,pose.inverse());
    instance->update_label(detection);
    instance->fast_update_centroid();
    instance->color_ = InstanceColorBar[instance->id_%InstanceColorBar.size()];
    instance_map.emplace(instance->id_,instance);

    return true;
}

bool SceneGraph::IsSemanticSimilar (const std::unordered_map<std::string,float> &measured_labels_a,
    const std::unordered_map<std::string,float> &measured_labels_b)
{
    if(measured_labels_a.size()<1 || measured_labels_b.size()<1) return false;

    for (const auto &label_score_a:measured_labels_a){
        for (const auto &label_score_b:measured_labels_b){
            if(label_score_a.first==label_score_b.first)return true;
        }
    }
    return false;
}

double SceneGraph::Compute2DIoU(
    const open3d::geometry::OrientedBoundingBox &box_a, const open3d::geometry::OrientedBoundingBox &box_b)
{
    auto box_a_aligned = box_a.GetAxisAlignedBoundingBox();
    auto box_b_aligned = box_b.GetAxisAlignedBoundingBox();

    // extract corners
    Eigen::Vector3d a0 = box_a_aligned.GetMinBound();
    Eigen::Vector3d a1 = box_a_aligned.GetMaxBound();
    Eigen::Vector3d b0 = box_b_aligned.GetMinBound();
    Eigen::Vector3d b1 = box_b_aligned.GetMaxBound();

    // find overlapped rectangle
    double x0 = std::max(a0(0),b0(0));
    double y0 = std::max(a0(1),b0(1));
    double x1 = std::min(a1(0),b1(0));
    double y1 = std::min(a1(1),b1(1));

    if(x0>x1 || y0>y1) return 0.0;

    // iou
    double intersection_area = ((x1-x0)*(y1-y0));
    double area_a = (a1(0)-a0(0))*(a1(1)-a0(1));
    double area_b = (b1(0)-b0(0))*(b1(1)-b0(1));
    double iou = intersection_area/(area_a+area_b-intersection_area);

    std::cout<<"floor iou: "<<iou<<std::endl;
    std::cout<<area_a<<","<<area_b<<","<<intersection_area<<std::endl;

    return iou;
}

double SceneGraph::Compute3DIoU (const O3d_Cloud_Ptr &cloud_a, const O3d_Cloud_Ptr &cloud_b, double inflation)
{
    auto vxgrid_a = open3d::geometry::VoxelGrid::CreateFromPointCloud(*cloud_a, inflation * config_.voxel_length);
    std::vector<bool> overlap = vxgrid_a->CheckIfIncluded(cloud_b->points_);
    double iou = double(std::count(overlap.begin(), overlap.end(), true)) / double(overlap.size());
    return iou;
}

void SceneGraph::merge_overlap_instances(std::vector<InstanceId> active_instances)
{
    double SEARCH_DISTANCE = 3.0; // in meters
    // double INFLATION_RATIO = 5.0;
    // double NMS_IOU = 0.25;

    std::vector<InstanceId> target_instances;
    if(active_instances.size()<1){
        for(auto &instance_j:instance_map)
            target_instances.emplace_back(instance_j.first);
    }
    else{
        target_instances = active_instances;
    }
    if(target_instances.size()<3) return;
    int old_instance_number = target_instances.size();

    // Find overlap instances
    open3d::utility::Timer timer;
    timer.Start();
    std::unordered_set<InstanceId> remove_instances;
    for(int i=0;i<target_instances.size();i++){
        auto instance_i = instance_map[target_instances[i]];
        std::string label_i = instance_i->get_predicted_class().first;
        std::cout<<instance_i->id_<<":"<<label_i<<";  "<<instance_i->point_cloud->points_.size()<<"\n";
        for(int j=i+1;j<target_instances.size();j++){
            if(remove_instances.find(target_instances[j])!=remove_instances.end()) 
                continue;
            
            auto instance_j = instance_map[target_instances[j]];
            double dist = (instance_i->centroid-instance_j->centroid).norm();

            if(!IsSemanticSimilar(instance_i->get_measured_labels(),instance_j->get_measured_labels())||
                dist>SEARCH_DISTANCE) continue;

            // Compute Spatial IoU
            InstancePtr large_instance, small_instance;
            if(instance_i->point_cloud->points_.size()>instance_j->point_cloud->points_.size()){
                large_instance = instance_i;
                small_instance = instance_j;
            }
            else{
                large_instance = instance_j;
                small_instance = instance_i;
            }
            
            double iou = Compute3DIoU(large_instance->point_cloud,small_instance->point_cloud,config_.merge_inflation);
            
            // Merge
            if(iou>config_.merge_iou){
                large_instance->merge_with(
                    small_instance->point_cloud,small_instance->get_measured_labels(),small_instance->get_observation_count());
                remove_instances.insert(small_instance->id_);
                // std::cout<<small_instance->id_<<" merged into "<<large_instance->id_<<std::endl;
                if(small_instance->id_==instance_i->id_) break;
            }   
        }
    }

    // Remove merged instances
    for(auto &instance_id:remove_instances){
        instance_map.erase(instance_id);
    }
    timer.Stop();

    o3d_utility::LogInfo("Merged {:d}/{:d} instances by 3D IoU. It takes {:f} ms.",
        remove_instances.size(),old_instance_number, timer.GetDurationInMillisecond());

}

void SceneGraph::merge_overlap_structural_instances()
{
    std::vector<InstanceId> target_instances;
    for(auto &instance_j:instance_map){
        if(instance_j.second->get_predicted_class().first=="floor")
            target_instances.emplace_back(instance_j.first);
    }

    if(target_instances.size()<2) return;

    int old_instance_number = target_instances.size();
    std::unordered_set<InstanceId> remove_instances;
    for(int i=0;i<target_instances.size();i++){
        auto instance_i = instance_map[target_instances[i]];
        std::string label_i = instance_i->get_predicted_class().first;
        for(int j=i+1;j<target_instances.size();j++){
            auto instance_j = instance_map[target_instances[j]];

            // Compute 2D IoU
            InstancePtr large_instance, small_instance;
            if(instance_i->point_cloud->points_.size()>instance_j->point_cloud->points_.size()){
                large_instance = instance_i;
                small_instance = instance_j;
            }
            else{
                large_instance = instance_j;
                small_instance = instance_i;
            }
            
            double iou = Compute2DIoU(*large_instance->min_box, *small_instance->min_box);
            
            // Merge
            if(iou>0.05){
                large_instance->merge_with(
                    small_instance->point_cloud,small_instance->get_measured_labels(),small_instance->get_observation_count());
                remove_instances.insert(small_instance->id_);
                // std::cout<<small_instance->id_<<" merged into "<<large_instance->id_<<std::endl;
                if(small_instance->id_==instance_i->id_) break;
            }   
        }
    }

    // remove merged instances
    for(auto &instance_id:remove_instances){
        instance_map.erase(instance_id);
    }

    o3d_utility::LogInfo("Merged {:d}/{:d} floor instances by 2D IoU.",remove_instances.size(),old_instance_number);

}

void SceneGraph::extract_bounding_boxes()
{
    // std::vector<std::shared_ptr<const open3d::geometry::Geometry>> viz_geometries;
    open3d::utility::Timer timer;
    timer.Start();
    int count = 0;

    for (const auto &instance: instance_map){
        instance.second->filter_pointcloud_statistic();
        // instance.second->filter_pointcloud_by_cluster();
        instance.second->CreateMinimalBoundingBox();
        if(instance.second->min_box->IsEmpty()||instance.second->point_cloud->points_.size()<config_.shape_min_points) continue;

        count ++;
    }
    timer.Stop();
    o3d_utility::LogInfo("Extract {:d} valid bounding box in {:f} ms",count,timer.GetDurationInMillisecond());

    // open3d::visualization::DrawGeometries(viz_geometries,"Open3d",1600,900);    
}

std::vector<std::shared_ptr<const open3d::geometry::Geometry>> SceneGraph::get_geometries(bool point_cloud, bool bbox)
{
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> viz_geometries;
    for (const auto &instance: instance_map){
        if(instance.second->point_cloud->points_.size()<config_.shape_min_points) continue;
        if(point_cloud){
            viz_geometries.emplace_back(instance.second->point_cloud);
        }
        if(bbox&&!instance.second->min_box->IsEmpty()){ 
            viz_geometries.emplace_back(instance.second->min_box);
        }
    }
    return viz_geometries;
}

void SceneGraph::Transform(const Eigen::Matrix4d &pose)
{
    for (const auto &instance: instance_map){
        instance.second->point_cloud->Transform(pose);
        instance.second->centroid = instance.second->point_cloud->GetCenter();
    }
}

bool SceneGraph::Save(const std::string &path)
{
    using namespace o3d_utility::filesystem;
    if(!DirectoryExists(path)) MakeDirectory(path);

    open3d::geometry::PointCloud global_instances_pcd;

    typedef std::pair<InstanceId,std::string> InstanceInfo;
    std::vector<InstanceInfo> instance_info;

    for (const auto &instance: instance_map){
        LabelScore semantic_class_score = instance.second->get_predicted_class();
        auto instance_cloud = instance.second->extract_point_cloud();
        if(instance_cloud->points_.size()<config_.min_instance_points) continue;

        global_instances_pcd += *instance_cloud;
        stringstream ss; // instance info string
        ss<<std::setw(4)<<std::setfill('0')<<instance.second->id_;
        open3d::io::WritePointCloud(path+"/"+ss.str()+".ply",*instance_cloud);

        ss<<";"
            <<semantic_class_score.first<<"("<<std::fixed<<std::setprecision(2)<<semantic_class_score.second<<")"<<";"
            <<instance.second->get_observation_count()<<";"
            <<instance.second->get_measured_labels_string()<<";"
            <<instance_cloud->points_.size()<<";\n";

        instance_info.emplace_back(instance.second->id_,ss.str());
        Eigen::Vector3d pt_centroid = instance_cloud->GetCenter();
        Eigen::Vector3d vl_centroid = instance.second->centroid;
        std::cout<<instance.second->id_<<":"<<pt_centroid.transpose()<<";  "<<vl_centroid.transpose()<<"\n";

        o3d_utility::LogInfo("Instance {:s} has {:d} points",semantic_class_score.first, instance_cloud->points_.size());
    }

    // Sort instance info and write it to text 
    std::ofstream ofs(path+"/instance_info.txt",std::ofstream::out);
    ofs<<"# instance_id;semantic_class(aggregate_score);observation_count;label_measurements;points_number\n";
    std::sort(instance_info.begin(),instance_info.end(),[](const InstanceInfo &a, const InstanceInfo &b){
        return a.first<b.first;
    });
    for (const auto &info:instance_info){
        ofs<<info.second;
    }
    ofs.close();

    // Save global instance map
    if(global_instances_pcd.points_.size()<1) return false;

    open3d::io::WritePointCloud(path+"/instance_map.ply",global_instances_pcd);
    o3d_utility::LogWarning("Save semantic instances to {:s}",path);

    return true;
}

bool SceneGraph::load(const std::string &path)
{
    o3d_utility::LogInfo("Load SceneGraph from {:s}",path);
    using namespace o3d_utility::filesystem;
    if(!DirectoryExists(path)) return false;

    // Load instance info
    std::ifstream ifs(path+"/instance_info.txt",std::ifstream::in);
    std::string line;
    std::getline(ifs,line); // skip header
    while(std::getline(ifs,line)){
        std::stringstream ss(line);
        std::string instance_id_str;
        std::getline(ss,instance_id_str,';');
        InstanceId instance_id = std::stoi(instance_id_str);
        std::string label_score_str, observ_str, label_measurments_str, observation_count_str;
        std::getline(ss,label_score_str,';');
        std::getline(ss,observ_str,';');
        std::getline(ss,label_measurments_str,';');
        // std::getline(ss,observation_count_str,')');
        
        InstancePtr instance_toadd = std::make_shared<Instance>(instance_id,10,instance_config);
        instance_toadd->load_previous_labels(label_measurments_str);
        instance_toadd->point_cloud = open3d::io::CreatePointCloudFromFile(path+"/"+instance_id_str+".ply");
        instance_toadd->centroid = instance_toadd->point_cloud->GetCenter();
        instance_toadd->color_ = InstanceColorBar[instance_id%InstanceColorBar.size()];
        instance_map.emplace(instance_id,instance_toadd);

        // cout<<instance_id_str<<","<<label_measurments_str
        //     <<","<<cloud->points_.size()
        //     <<"\n";
    
    }

    o3d_utility::LogInfo("Load {:d} instances",instance_map.size());

    return true;

}

}