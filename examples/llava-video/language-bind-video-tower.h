#ifndef LANGUAGE_BIND_VIDEO_TOWER_H
#define LANGUAGE_BIND_VIDEO_TOWER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Function declarations
std::vector<cv::Mat> loadAndExtractFrames(const std::string& videoPath, int numFrames);
void preprocessFrames(std::vector<cv::Mat>& frames, bool enableRandomFlip = false);

#endif // LANGUAGE_BIND_VIDEO_TOWER_H
