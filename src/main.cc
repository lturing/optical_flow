#include<iostream>
#include<fstream>
#include<chrono>
#include<vector>
#include<opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "pvio.h"
#include "frame.h"
#include "yaml_config.h"
#include "opencv_image.h"
#include "track.h"
#include "poisson_disk_filter.h"

using namespace std;
using namespace pvio;

void LoadImages(const string &strImagePath, std::vector<string> &vstrImages);



int main(int argc, const char *argv[]) {

    if(argc != 3)
    {
        cerr << endl << "Usage: ./optical_flow path_to_settings path_to_sequence_folder_1" << endl;
        return 1;
    }

    std::vector<string> vstrImageFilenames;
    string strImageDir = argv[2];
    LoadImages(strImageDir, vstrImageFilenames);

    std::shared_ptr<pvio::Config> config = std::make_shared<pvio::extra::YamlConfig>(argv[1]);
    config->log_config();
    std::shared_ptr<pvio::Frame> frame;
    std::shared_ptr<pvio::Frame> last_frame = nullptr;
    
    std::vector<std::shared_ptr<pvio::Frame>> frames;

    if (vstrImageFilenames.size() > 1)
    {
        bool is_first = true;
        cv::Mat im;
    
        for (int i = 0; i < vstrImageFilenames.size(); i++)
        {
            //cout << vstrImageFilenames[i] << endl;
            //im = cv::imread(vstrImageFilenames[i],cv::IMREAD_GRAYSCALE); //cv::IMREAD_UNCHANGED); 
            cv::Mat img_distorted = cv::imread(vstrImageFilenames[i], cv::IMREAD_GRAYSCALE);

            cv::Mat dist_coeffs = (cv::Mat_<float>(4, 1) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);
            cv::Mat K = (cv::Mat_<float>(3, 3) << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1);
            cv::undistort(img_distorted, im, K, dist_coeffs);

            std::shared_ptr<pvio::extra::OpenCvImage> opencv_image = std::make_shared<pvio::extra::OpenCvImage>();
            opencv_image->image = im;
            opencv_image->preprocess();

            frame = std::make_shared<Frame>();
            frame->image = opencv_image;
            
            if (frames.size() > 0)
            {
                last_frame = frames[frames.size()-1];
                std::vector<pvio::vector<2>> next_keypoints; 
                std::vector<char> status;
                last_frame->image->track_keypoints(frame->image.get(), last_frame->keypoints, next_keypoints, status);

                /*
                // filter keypoints based on track length
                std::vector<std::pair<size_t, size_t>> keypoint_index_track_length;
                keypoint_index_track_length.reserve(last_frame->keypoints.size());
                for (size_t i = 0; i < last_frame->keypoints.size(); ++i) {
                    if (status[i] == 0) continue;
                    Track *track = get_track(i);
                    if (track == nullptr) continue;
                    keypoint_index_track_length.emplace_back(i, track->keypoint_num());
                }

                std::sort(keypoint_index_track_length.begin(), keypoint_index_track_length.end(), [](const auto &a, const auto &b) {
                    return a.second > b.second;
                });

                PoissonDiskFilter<2> filter(config->feature_tracker_min_keypoint_distance());
                for (auto &[keypoint_index, track_length] : keypoint_index_track_length) {
                    vector<2> pt = next_keypoints[keypoint_index];
                    if (filter.permit_point(pt)) {
                        filter.preset_point(pt);
                    } else {
                        status[keypoint_index] = 0;
                    }
                }
                */

                for (size_t curr_keypoint_index = 0; curr_keypoint_index < last_frame->keypoints.size(); ++curr_keypoint_index) {
                    if (status[curr_keypoint_index]) {
                        size_t next_keypoint_index = frame->keypoint_num();
                        frame->append_keypoint(next_keypoints[curr_keypoint_index]);
                        if (last_frame->tracks[curr_keypoint_index] == nullptr) {
                            std::shared_ptr<Track> track = std::make_shared<Track>();
                            last_frame->tracks[curr_keypoint_index] = track;
                            frame->tracks[next_keypoint_index] = track;
                            track->frames.push_back(last_frame);
                            track->frames.push_back(frame);

                            track->keypoint_refs[last_frame] = curr_keypoint_index;
                            track->keypoint_refs[frame] = next_keypoint_index;

                        }
                        else 
                        {
                            frame->tracks[next_keypoint_index] = last_frame->tracks[curr_keypoint_index];
                            last_frame->tracks[curr_keypoint_index]->keypoint_refs[frame] = next_keypoint_index;
                            last_frame->tracks[curr_keypoint_index]->frames.push_back(frame);
                        }
                    }
                }

            }

            frame->image->detect_keypoints(frame->keypoints, config->feature_tracker_max_keypoint_detection(), config->feature_tracker_min_keypoint_distance());
            frame->tracks.resize(frame->keypoints.size(), nullptr);

            frames.push_back(frame);

            cv::Mat nim;
            cv::cvtColor(im, nim, cv::COLOR_GRAY2BGR);


            for (int i = 0; i < frame->keypoints.size(); i++)
            {
                cv::Scalar standardColor(0, 0, 255); //h & 0xFF, (h >> 4) & 0xFF, (h >> 8) & 0xFF);
                //cv::Scalar standardColor(255, 0, 0);
                pvio::vector<2> p = frame->keypoints[i];
                cv::circle(nim,cv::Point2f(p[0], p[1]),2,standardColor,-1);
            }

            for (auto track : frame->tracks)
            {
                if(track)
                {
                    size_t h = track->id() * 6364136223846793005u + 1442695040888963407;

                    cv::Scalar standardColor(h & 0xFF, (h >> 4) & 0xFF, (h >> 8) & 0xFF);
                    
                    for (int i = track->frames.size() - 1; i > 0; i--){
                        int j = i - 1;
                        pvio::vector<2> pi = track->frames[i]->keypoints[track->keypoint_refs[track->frames[i]]];
                        pvio::vector<2> pj = track->frames[j]->keypoints[track->keypoint_refs[track->frames[j]]];
                                                    
                        cv::Point2f pt1 = cv::Point2f(pi[0], pi[1]);
                        cv::Point2f pt2 = cv::Point2f(pj[0], pj[1]);

                        //std::cout << pi.transpose() << ", " << pj.transpose() << std::endl;
                        
                        cv::line(nim, pt1, pt2, standardColor, 2);
                    }
                }
            }
            cv::imshow("matches", nim);
            cv::waitKey(1);
            
        }
    }


    return 0;
}


void LoadImages(const string &strImageDir, std::vector<string> &vstrImages) 
{
    FILE *fhandler = fopen((strImageDir + "/data.csv").c_str(), "r");
    vstrImages.reserve(5000);

    char header_line[2048];
    fscanf(fhandler, "%2047[^\r]\r\n", header_line);
    char filename_buffer[2048];
    double t;
    while(!feof(fhandler))
    {
        memset(filename_buffer, 0, 2048);
        if (fscanf(fhandler, "%lf,%2047[^\r]\r\n", &t, filename_buffer) != 2) {
            break;
        }
        vstrImages.push_back(strImageDir + "/data/" + std::string(filename_buffer));
    }
    fclose(fhandler);
}


