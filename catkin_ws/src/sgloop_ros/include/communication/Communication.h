#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <map>
#include <fstream>
#include <torch/torch.h>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sgloop_ros/CoarseGraph.h>
#include <sgloop_ros/DenseGraph.h>
#include <eigen3/Eigen/Dense>

namespace SgCom
{
    struct Log
    {
        int frame_id;
        float timestamp;
        std::string direction; // "pub" or "sub"
        std::string msg_type; // "CoarseGraph", "DenseGraph"
        bool checksum;
    };
    
    struct AgentDataDict{
        int frame_id=-1;
        float received_timestamp;
        int N=0;
        int X=0; // number of points
        int D=0; // coarse node feature dimension
        std::vector<uint32_t> nodes; // (N,)
        std::vector<uint32_t> instances; // (N,)
        std::vector<Eigen::Vector3d> centroids; // (N,3)
        std::vector<std::vector<float>> features_vec; // (N,D)
        std::vector<Eigen::Vector3d> xyz; // (X,3)
        std::vector<uint32_t> labels; // (X,) node ids, each label in [0,N)

        void clear(){
            nodes.clear();
            instances.clear();
            centroids.clear();
            features_vec.clear();
            xyz.clear();
            labels.clear();
            N=0;
            X=0;
            D=0;
        };

        std::string print_msg()const{
            std::stringstream ss;
            for(int i=0;i<N;i++){
                ss<<"instance: "<<instances[i]<<" centroid: "<<centroids[i].transpose();
                ss<<"\n";
            }
            return ss.str();
        }

    };

    void convert_tensor(const std::vector<std::vector<float>> &features_vector, 
                    const int &N, const int &D, torch::Tensor &features_tensor);
                    
    class Communication
    {
        public:
            /// @brief Each MASTER_AGENT publish their features to "MASTER_AGENT/graph_topics"
            ///        And it subscribes to "OTHER_AGENT/graph_topics" for other agents' features.
            /// @param nh 
            /// @param nh_private 
            /// @param master_agent 
            /// @param other_agents 
            Communication(ros::NodeHandle &nh, ros::NodeHandle &nh_private,
                        std::string master_agent="agent_a", std::vector<std::string> other_agents={});
            ~Communication(){};

            bool broadcast_coarse_graph(int frame_id,
                                const std::vector<uint32_t> &instances,
                                const std::vector<Eigen::Vector3d> &centroids,
                                const int& N, const int &D,
                                const std::vector<std::vector<float>> &features);


            /// @brief  Broadcast dense graph to other agents.
            /// @param frame_id
            /// @param nodes (N,)
            /// @param instances (N,)
            /// @param centroids (N,3)
            /// @param features (N,D) coarse node features
            /// @param xyz (X,3) global point cloud
            /// @param labels (X,) node ids, each label in [0,N)
            bool broadcast_dense_graph(int frame_id,
                                    const std::vector<uint32_t> &nodes,
                                    const std::vector<uint32_t> &instances,
                                    const std::vector<Eigen::Vector3d> &centroids,
                                    const std::vector<std::vector<float>> &features,
                                    const std::vector<Eigen::Vector3d> &xyz,
                                    const std::vector<uint32_t> &labels);

            void coarse_graph_callback(const sgloop_ros::CoarseGraph::ConstPtr &msg,std::string agent_name);

            void dense_graph_callback(const sgloop_ros::DenseGraph::ConstPtr &msg,std::string agent_name);

            const AgentDataDict& get_remote_agent_data(const std::string agent_name)const;
            // AgentDataDict &queried_data_dict)const;

            bool write_logs(const std::string &out_dir);

        private:

            float frame_duration; // in seconds
            float time_offset; // in seconds

            ros::Publisher pub_coarse_sg, pub_dense_sg;
            std::vector<ros::Subscriber> agents_subscriber;
            std::vector<ros::Subscriber> agents_dense_subscriber;

            std::string local_agent;
            std::map<std::string,AgentDataDict> agents_data_dict;

            std::vector<Log> pub_logs;
            std::vector<Log> sub_logs;

            int last_exchange_frame_id; // At an exchange frame, the latest agent data dict is exported.

    };
    
    

}


#endif // COMMUNICATION_H
