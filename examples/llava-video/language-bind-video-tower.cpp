#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include "language-bind-video-tower.h"


/*std::vector<cv::Mat> loadAndExtractFrames(const std::string& videoPath, int numFrames) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << videoPath << std::endl;
        return {};
    }

    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::vector<int> frameIndices;
    double step = totalFrames / static_cast<double>(numFrames);
    for (int i = 0; i < numFrames; ++i) {
        frameIndices.push_back(static_cast<int>(std::round(i * step)));
    }

    std::vector<cv::Mat> frames;
    cv::Mat frame;
    for (int idx : frameIndices) {
        cap.set(cv::CAP_PROP_POS_FRAMES, idx);
        if (!cap.read(frame)) {
            std::cerr << "Error reading frame " << idx << std::endl;
            continue;
        }
        frames.push_back(frame.clone());
    }

    return frames;
}*/

// Custom linspace function for C++
std::vector<int> linspace(int start, int end, int num) {
    std::vector<int> linspaced;
    double delta = (end - start) / static_cast<double>(num - 1);

    for(int i = 0; i < num - 1; ++i) {
        linspaced.push_back(static_cast<int>(start + delta * i));
    }
    linspaced.push_back(end); // Ensure that end is always included

    return linspaced;
}

std::vector<cv::Mat> loadAndExtractFrames(const std::string& videoPath, int numFrames) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return {};
    }

    int totalFrames = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::vector<int> frameIndices = linspace(0, totalFrames - 1, numFrames);
    std::vector<cv::Mat> frames;

    for (int idx : frameIndices) {
        cap.set(cv::CAP_PROP_POS_FRAMES, idx);
        cv::Mat frame;
        if (cap.read(frame)) {
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            frames.push_back(frame);
        }
    }

    return frames;
}

const cv::Scalar OPENAI_DATASET_MEAN = cv::Scalar(0.48145466, 0.4578275, 0.40821073);
const cv::Scalar OPENAI_DATASET_STD = cv::Scalar(0.26862954, 0.26130258, 0.27577711);

void preprocessFrames(std::vector<cv::Mat>& frames, bool enableRandomFlip) {
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);


    for (auto& frame : frames) {
        // Convert to float and normalize
        frame.convertTo(frame, CV_32FC3, 1.0 / 255.0);
        frame = (frame - OPENAI_DATASET_MEAN) / OPENAI_DATASET_STD;

        // Resize and center crop to 224x224
        int shorterSide = std::min(frame.rows, frame.cols);
        cv::resize(frame, frame, cv::Size(224 * frame.cols / shorterSide, 224 * frame.rows / shorterSide));
        int cropX = (frame.cols - 224) / 2;
        int cropY = (frame.rows - 224) / 2;
        cv::Rect roi(cropX, cropY, 224, 224);
        frame = frame(roi);

        // Random horizontal flip
        if (enableRandomFlip && dis(gen) > 0.5) {
            cv::flip(frame, frame, 1);
        }
    }
}
