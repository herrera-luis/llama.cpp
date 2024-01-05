// VideoEmbedding.h

#ifndef VIDEO_EMBEDDING_H
#define VIDEO_EMBEDDING_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "clip.h"


// Forward declaration of clip_ctx
struct clip_ctx;

// Structure to store video embeddings
struct llava_video_embed {
    std::vector<float*> embeddings; // A vector to store embeddings for each frame
  // std::vector<std::unique_ptr<uint8_t[]>> embeddings;
    int n_frames;                   // The number of frames in the video
};

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

// Function declarations
bool process_frame_and_get_embedding(clip_ctx * ctx_clip, int n_threads, const cv::Mat& frame, float * frame_embed);
bool convert_mat_to_clip_image_u8(const cv::Mat& mat, clip_image_u8 * img);
llava_video_embed * init_video_embed(int n_frames, int embedding_size);
void free_video_embed(llava_video_embed * video_embed);
bool encode_frame_with_clip(clip_ctx * ctx_clip, clip_image_u8 * img, float * embedding);
// std::vector<cv::Mat> extractFramesFromVideoBytes(const std::vector<unsigned char>& videoBytes, int numFrames);
std::vector<float*> generateFrameEmbeddings(clip_ctx * ctx_clip, const std::vector<cv::Mat>& frames);
clip_image_f32* preprocess_clip_image(clip_ctx* ctx_clip, clip_image_u8* img);

#endif // VIDEO_EMBEDDING_H
