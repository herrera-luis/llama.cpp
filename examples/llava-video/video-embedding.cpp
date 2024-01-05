#include "video-embedding.h"
#include "clip.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>




// Function to process a single frame and get its embedding
bool process_frame_and_get_embedding(clip_ctx * ctx_clip, int n_threads, const cv::Mat& frame, float * frame_embed) {
    // Convert cv::Mat frame to clip_image_u8
    clip_image_u8 * img = make_clip_image_u8();
    if (!convert_mat_to_clip_image_u8(frame, img)) {
        clip_image_u8_free(img);
        return false;
    }

    // Preprocess the image
    clip_image_f32 * img_res = make_clip_image_f32();
    if (!clip_image_preprocess(ctx_clip, img, img_res, true)) { // Assuming padding to square is required
        clip_image_f32_free(img_res);
        clip_image_u8_free(img);
        return false;
    }

    // Encode the image to get the embedding
    if (!clip_image_encode(ctx_clip, n_threads, img_res, frame_embed)) {
        clip_image_f32_free(img_res);
        clip_image_u8_free(img);
        return false;
    }

    // Clean up
    clip_image_f32_free(img_res);
    clip_image_u8_free(img);

    return true;
}

// Helper function to convert cv::Mat to clip_image_u8
bool convert_mat_to_clip_image_u8(const cv::Mat& mat, clip_image_u8 * img) {
    if (mat.empty() || img == nullptr) {
        return false;
    }

    if (mat.type() == CV_8UC3) {
        std::cerr << " it's type CV_8UC3" << std::endl;
        memcpy(img->data, mat.data, img->size);
    } else if (mat.type() == CV_8UC1) {
        std::cerr << " it's type CV_8UC1" << std::endl;
        // Convert grayscale to RGB
        for (int y = 0; y < mat.rows; ++y) {
            for (int x = 0; x < mat.cols; ++x) {
                uint8_t val = mat.at<uint8_t>(y, x);
                img->data[(y * mat.cols + x) * 3 + 0] = val;
                img->data[(y * mat.cols + x) * 3 + 1] = val;
                img->data[(y * mat.cols + x) * 3 + 2] = val;
            }
        }
    } else if (mat.type() == CV_8UC4) {
        // Convert RGBA to RGB by dropping the alpha channel
        std::cerr << "it's type CV_8UC4" << std::endl;
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);

    } /*else {
        std::cerr << "Unsupported image type: " << mat.type() << std::endl;
        return false;
    }*/

    // Assuming mat is now always CV_8UC3 due to the conversion above
    img->nx = mat.cols;
    img->ny = mat.rows;
    img->size = mat.cols * mat.rows * 3;
    img->data = new uint8_t[img->size];
    std::cerr << "Frame converted to uint8_t" << std::endl;
    return true;
}



/*bool convert_mat_to_clip_image_u8(const cv::Mat& mat, clip_image_u8 * img) {
    if (mat.empty() || img == nullptr) {
        return false;
    }

    cv::Mat convertedMat;
    if (mat.type() == CV_8UC3) {
        convertedMat = mat; // Directly use the original mat
    } else if (mat.type() == CV_8UC1) {
        // Convert grayscale to RGB
        cv::cvtColor(mat, convertedMat, cv::COLOR_GRAY2BGR);
    } else if (mat.type() == CV_8UC4) {
        // Convert RGBA to RGB by dropping the alpha channel
        cv::cvtColor(mat, convertedMat, cv::COLOR_RGBA2BGR);
    } else {
        std::cerr << "Unsupported image type: " << mat.type() << std::endl;
        return false;
    }

    // Now convertedMat is guaranteed to be CV_8UC3
    img->nx = convertedMat.cols;
    img->ny = convertedMat.rows;
    img->size = convertedMat.cols * convertedMat.rows * 3;
    img->data = new uint8_t[img->size];
    memcpy(img->data, convertedMat.data, img->size);

    return true;
}*/

/*llava_video_embed * init_video_embed(int n_frames, int embedding_size) {


    llava_video_embed * video_embed = new llava_video_embed;
    video_embed->n_frames = n_frames;
    video_embed->embeddings.reserve(n_frames);

    for (int i = 0; i < n_frames; ++i) {
        // Example: Suppose each embedding is of size 1024
        const size_t size_of_embedding = 512;
     //   float* frame_embedding = new float[embedding_size];
         std::unique_ptr<uint8_t[]> frame_embedding(new uint8_t[size_of_embedding]);

      // video_embed->embeddings.push_back(frame_embedding);
      // ... populate frame_embedding ...
        video_embed->embeddings.push_back(std::move(frame_embedding));
    }

    return video_embed;
}*/

llava_video_embed * init_video_embed(int n_frames, int embedding_size) {
    llava_video_embed * video_embed = new llava_video_embed;
    video_embed->n_frames = n_frames;
    video_embed->embeddings.reserve(n_frames);

    for (int i = 0; i < n_frames; ++i) {
        float* frame_embedding = new float[embedding_size];
        video_embed->embeddings.push_back(frame_embedding);
    }

    return video_embed;
}


void free_video_embed(llava_video_embed * video_embed) {
    if (video_embed) {
        for (auto& embedding : video_embed->embeddings) {
            delete[] embedding;
        }
        delete video_embed;
    }
}

// Function to encode a single frame with CLIP
bool encode_frame_with_clip(clip_ctx * ctx_clip, clip_image_u8 * img, float * embedding) {
    // Check for null pointers
    if (!ctx_clip || !img || !embedding) {
        fprintf(stderr, "Invalid input to encode_frame_with_clip\n");
        return false;
    }

    // Preprocess the image (if required by your CLIP model)
    clip_image_f32 * preprocessed_img = preprocess_clip_image(ctx_clip, img);
    if (!preprocessed_img) {
        fprintf(stderr, "Failed to preprocess image\n");
        return false;
    }

    const int n_threads = 8; /* your desired number of threads */;

    // Encode the image using CLIP
    /*int n_img_pos;
    if (!clip_image_encode(ctx_clip, preprocessed_img, embedding, &n_img_pos)) {
        fprintf(stderr, "Failed to encode image with CLIP\n");
        clip_image_f32_free(preprocessed_img);
        return false;
    }*/
     if (!clip_image_encode(ctx_clip, n_threads, preprocessed_img, embedding)) {
        fprintf(stderr, "Failed to encode image with CLIP\n");
        clip_image_f32_free(preprocessed_img);
        return false;
    }

    // Clean up
    clip_image_f32_free(preprocessed_img);
    return true;
}

// Function to preprocess an image for CLIP model
clip_image_f32* preprocess_clip_image(clip_ctx* ctx_clip, clip_image_u8* img) {
    if (!ctx_clip || !img) {
        fprintf(stderr, "Invalid input to preprocess_clip_image\n");
        return nullptr;
    }

    // Create a new clip_image_f32 object for the preprocessed image
    clip_image_f32* preprocessed_img = new clip_image_f32;
    if (!preprocessed_img) {
        fprintf(stderr, "Failed to allocate memory for preprocessed image\n");
        return nullptr;
    }

    // Example preprocessing steps:
    // 1. Resize the image to the required input size of the CLIP model
    // 2. Normalize pixel values (e.g., scaling to [0, 1] range or mean subtraction)

    // Assuming the CLIP model requires a specific image size (e.g., 224x224)
    int target_width = 224;
    int target_height = 224;
    cv::Mat cv_img(img->ny, img->nx, CV_8UC3, img->data);
    cv::resize(cv_img, cv_img, cv::Size(target_width, target_height));

    // Convert to float and normalize
    cv_img.convertTo(cv_img, CV_32FC3, 1.0 / 255.0);
    // Assuming normalization with mean and std (update with actual values)
    cv::Scalar mean = {0.485, 0.456, 0.406};
    cv::Scalar std = {0.229, 0.224, 0.225};
    cv_img = (cv_img - mean) / std;

    // Copy the processed data to preprocessed_img
    preprocessed_img->nx = target_width;
    preprocessed_img->ny = target_height;
    preprocessed_img->size = target_width * target_height * 3;
    preprocessed_img->data = new float[preprocessed_img->size];
    std::memcpy(preprocessed_img->data, cv_img.data, preprocessed_img->size * sizeof(float));

    return preprocessed_img;
}


/*std::vector<cv::Mat> extractFramesFromVideoBytes(const std::vector<unsigned char>& videoBytes, int numFrames) {
    // Write the video bytes to a temporary file
    std::string tempFilePath = "temp_video.mp4"; // You can generate a unique file name if needed
    std::ofstream outFile(tempFilePath, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(videoBytes.data()), videoBytes.size());
    outFile.close();

    // Use the temporary file to create a VideoCapture object
    cv::VideoCapture cap(tempFilePath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return {};
    }

    // Calculate the step size to evenly sample frames throughout the video
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double step = totalFrames / static_cast<double>(numFrames);

    std::vector<cv::Mat> frames;
    cv::Mat frame;

    // Extract frames
    for (int i = 0; i < numFrames; ++i) {
        int frameIndex = static_cast<int>(std::round(i * step));
        cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);

        if (!cap.read(frame)) {
            std::cerr << "Error: Could not read frame " << frameIndex << std::endl;
            continue;
        }

        frames.push_back(frame.clone());
    }

    // Clean up: Delete the temporary file
    std::remove(tempFilePath.c_str());

    return frames;
}*/

std::vector<float*> generateFrameEmbeddings(clip_ctx* ctx_clip, int n_threads, const std::vector<cv::Mat>& frames) {
    std::vector<float*> embeddings;
    int embedding_size = clip_n_mmproj_embd(ctx_clip); // Assuming this function returns the size of the embedding

    // For each frame, generate an embedding
    for (const auto& frame : frames) {
        float* embedding = new float[embedding_size];
        if (!process_frame_and_get_embedding(ctx_clip, n_threads, frame, embedding)) {
            // Handle error: could be logging or cleaning up
            delete[] embedding;
            continue;
        }
        embeddings.push_back(embedding);
    }
    return embeddings;
}
