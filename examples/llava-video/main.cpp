#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

std::vector<cv::Mat> loadAndTransformVideo(
    const std::string& videoPath,
    int numFrames,
    double clipStartSec = 0.0,
    double clipEndSec = -1.0 // Use -1 to indicate full duration
) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return {};
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double duration = totalFrames / fps;

    int startFrame = static_cast<int>(clipStartSec * fps);
    int endFrame = clipEndSec < 0 ? totalFrames : static_cast<int>(clipEndSec * fps);
    endFrame = std::min(endFrame, totalFrames);

    std::vector<int> frameIndices;
    double step = (endFrame - startFrame) / static_cast<double>(numFrames);
    for (int i = 0; i < numFrames; ++i) {
        frameIndices.push_back(startFrame + static_cast<int>(std::round(i * step)));
    }

    std::vector<cv::Mat> frames;
    cv::Mat frame;
    for (int idx : frameIndices) {
        cap.set(cv::CAP_PROP_POS_FRAMES, idx);
        if (!cap.read(frame)) {
            std::cerr << "Error reading frame " << idx << std::endl;
            continue;
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        frames.push_back(frame.clone());
    }

    return frames;
}

int main() {
    std::string videoPath = "./media/sample_demo_1.mp4";
    int numFrames = 8; // Number of frames to extract
    double clipStartSec = 0.0; // Start of the clip in seconds
    double clipEndSec = -1.0; // End of the clip in seconds (-1 for full duration)

    std::vector<cv::Mat> frames = loadAndTransformVideo(videoPath, numFrames, clipStartSec, clipEndSec);

    return 1;
}
