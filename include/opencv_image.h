
#ifndef PVIO_EXTRA_OPENCV_IMAGE_H
#define PVIO_EXTRA_OPENCV_IMAGE_H

#include <opencv2/opencv.hpp>

#include "pvio.h"

namespace pvio::extra {

class OpenCvImage : public Image {
  public:
    OpenCvImage();

    size_t width() const override {
        return image.cols;
    }

    size_t height() const override {
        return image.rows;
    }

    size_t level_num() const override {
        return 3;
    }

    void detect_keypoints(std::vector<vector<2>> &keypoints, size_t max_points = 0, double keypoint_distance = 0.5) const override;
    void track_keypoints(const Image *next_image, const std::vector<vector<2>> &curr_keypoints, std::vector<vector<2>> &next_keypoints, std::vector<char> &result_status) const override;

    void preprocess() override;
    void correct_distortion(const matrix<3> &intrinsics, const vector<4> &coeffs);

    cv::Mat image;

    std::vector<cv::Mat> image_pyramid;
    std::vector<cv::Mat> image_levels;
    std::vector<vector<2>> scale_levels;

    static cv::CLAHE *clahe();
    static cv::GFTTDetector *gftt();
    static cv::FastFeatureDetector *fast();
    static cv::ORB *orb();
};

} // namespace pvio::extra

#endif /* PVIO_EXTRA_OPENCV_IMAGE_H */
