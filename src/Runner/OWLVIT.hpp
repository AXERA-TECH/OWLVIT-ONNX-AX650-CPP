#pragma once
#include <map>
#include "vector"
#include <string>
#include "fstream"
#include "thread"

#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"

#include "sample_log.h"
#include "Tokenizer.hpp"

class OWLVIT
{
protected:
    std::string device{"cpu"};
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> TextEncoderSession, DecoderSession;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    const char
        *TextEncInputNames[2]{"input_ids", "attention_mask"},
        *TextEncOutputNames[1]{"text_embeds"},
        *DecoderInputNames[3]{"image_embeds", "/owlvit/Div_output_0", "input_ids"},
        *DecoderOutputNames[1]{"logits"};

    float _mean_val[3] = {122.7709383, 116.7460125, 104.09373615};
    float _std_val[3] = {1 / 68.5005327, 1 / 66.6321579, 1 / 70.32316305};
    std::shared_ptr<TokenizerBase> tokenizer;

    // std::vector<float> image_features_input;
    // std::vector<float> text_features_input;
    // std::vector<float> pred_boxes;
    // std::vector<int64> input_ids;

    std::vector<int64_t> image_features_shape;
    std::vector<int64_t> text_features_shape = {1, 512};

    int LEN_IMAGE_FEATURE = 24 * 24 * 768;
    int CNT_PRED_BOXES = 576;
    int LEN_TEXT_FEATURE = 512;
    int LEN_TEXT_TOKEN = 16;
    int input_height = 768, input_width = 768;

    cv::Mat mat_input;

    static void get_input_data_letterbox(cv::Mat mat, cv::Mat &img_new, bool bgr2rgb = false)
    {
        /* letterbox process to support different letterbox size */
        int letterbox_rows = img_new.rows;
        int letterbox_cols = img_new.cols;
        float scale_letterbox;
        int resize_rows;
        int resize_cols;
        if ((letterbox_rows * 1.0 / mat.rows) < (letterbox_cols * 1.0 / mat.cols))
        {
            scale_letterbox = (float)letterbox_rows * 1.0f / (float)mat.rows;
        }
        else
        {
            scale_letterbox = (float)letterbox_cols * 1.0f / (float)mat.cols;
        }
        resize_cols = int(scale_letterbox * (float)mat.cols);
        resize_rows = int(scale_letterbox * (float)mat.rows);

        cv::resize(mat, mat, cv::Size(resize_cols, resize_rows));

        int top = (letterbox_rows - resize_rows) / 2;
        int bot = (letterbox_rows - resize_rows + 1) / 2;
        int left = (letterbox_cols - resize_cols) / 2;
        int right = (letterbox_cols - resize_cols + 1) / 2;

        // Letterbox filling
        cv::copyMakeBorder(mat, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        if (bgr2rgb)
        {
            cv::cvtColor(img_new, img_new, cv::COLOR_BGR2RGB);
        }
    }

public:
    OWLVIT()
    {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "owlvit");
        session_options = Ort::SessionOptions();
        session_options.SetInterOpNumThreads(4);
        session_options.SetIntraOpNumThreads(4);
        // 设置图像优化级别
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

    int get_image_feature_size()
    {
        return LEN_IMAGE_FEATURE;
    }

    int get_text_feature_size()
    {
        return LEN_TEXT_FEATURE;
    }

    bool load_tokenizer(std::string vocab_path)
    {
        tokenizer.reset(new TokenizerClip);
        // ALOGI("text token len %d", LEN_TEXT_TOKEN);
        // text_tokens_input = std::vector<int>(LEN_TEXT_TOKEN);
        return tokenizer->load_tokenize(vocab_path);
    }

    bool load_decoder(std::string decoder_path)
    {
        DecoderSession.reset(new Ort::Session(env, decoder_path.c_str(), session_options));
        // if (DecoderSession->GetInputCount() != 2 || DecoderSession->GetOutputCount() != 2)
        // {
        //     ALOGE("Model not loaded (invalid input/output count)");
        //     return false;
        // }
        return true;
    }

    bool load_text_encoder(std::string encoder_path)
    {
        TextEncoderSession.reset(new Ort::Session(env, encoder_path.c_str(), session_options));
        
        // if (TextEncoderSession->GetInputCount() != 1 || TextEncoderSession->GetOutputCount() != 1)
        // {
        //     ALOGE("Model not loaded (invalid input/output count)");
        //     return false;
        // }
        auto shape = TextEncoderSession->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        LEN_TEXT_FEATURE = 1;
        // text_features_shape.clear();
        for (size_t i = 0; i < shape.size(); i++)
        {
            LEN_TEXT_FEATURE *= shape[i];
            // text_features_shape.push_back(shape[i]);
        }
        ALOGI("text feature len %d", LEN_TEXT_FEATURE);
        return true;
    }

    virtual bool load_image_encoder(std::string encoder_path) = 0;
    virtual void encode(cv::Mat image, std::vector<float> &image_features, std::vector<cv::Rect2f> &pred_boxes) = 0;

    void encode(std::vector<std::string> &texts, std::vector<std::vector<int64>> &input_ids, std::vector<std::vector<float>> &text_features)
    {
        input_ids.resize(texts.size());
        text_features.resize(texts.size());
        Ort::RunOptions runOptions;
        for (size_t i = 0; i < texts.size(); i++)
        {
            // std::vector<int64> input_ids;
            tokenizer->encode_text(texts[i], input_ids[i]);
            std::vector<int64> attention_mask;
            for (size_t j = 0; j < input_ids[i].size(); j++)
            {
                attention_mask.push_back(1);
            }
            for (size_t j = input_ids[i].size(); j < LEN_TEXT_TOKEN; j++)
            {
                attention_mask.push_back(0);
                input_ids[i].push_back(0);
            }

            // print input_ids and attention_mask
            // for (size_t j = 0; j < input_ids[i].size(); j++)
            // {
            //     printf("%d ", input_ids[i][j]);
            // }
            // printf("\n");
            // for (size_t j = 0; j < attention_mask.size(); j++)
            // {
            //     printf("%d ", attention_mask[j]);
            // }
            // printf("\n");

            std::vector<Ort::Value> inputTensors;

            std::vector<int64_t> input_ids_shape = {1, LEN_TEXT_TOKEN};
            inputTensors.push_back((Ort::Value::CreateTensor<int64>(
                memory_info_handler, input_ids[i].data(), input_ids[i].size(), input_ids_shape.data(), input_ids_shape.size())));

            // std::vector<int64_t> attention_mask_shape = {1, LEN_TEXT_TOKEN};
            inputTensors.push_back((Ort::Value::CreateTensor<int64>(
                memory_info_handler, attention_mask.data(), attention_mask.size(), input_ids_shape.data(), input_ids_shape.size())));

            auto OutputTensors = TextEncoderSession->Run(runOptions, TextEncInputNames, inputTensors.data(),
                                                         inputTensors.size(), TextEncOutputNames, 1);

            auto &text_features_tensor = OutputTensors[0];
            auto text_features_tensor_ptr = text_features_tensor.GetTensorMutableData<float>();
            text_features[i].resize(LEN_TEXT_FEATURE);
            memcpy(text_features[i].data(), text_features_tensor_ptr, LEN_TEXT_FEATURE * sizeof(float));

            // print text features

            // for (size_t j = 0; j < text_features[i].size(); j++)
            // {
            //     printf("%f ", text_features[i][j]);
            // }
            // printf("\n");
        }
    }

    void decode(std::vector<float> &image_features, std::vector<float> &text_features, std::vector<int64> &input_ids, std::vector<float> &logits)
    {
        std::vector<Ort::Value> inputTensors;

        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, image_features.data(), LEN_IMAGE_FEATURE, image_features_shape.data(), image_features_shape.size()));
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, text_features.data(), LEN_TEXT_FEATURE, text_features_shape.data(), text_features_shape.size()));

        // for (size_t j = 0; j < input_ids.size(); j++)
        // {
        //     printf("%d ", input_ids[j]);
        // }
        // printf("\n");
        std::vector<int64_t> input_ids_shape = {1, LEN_TEXT_TOKEN};
        inputTensors.push_back((Ort::Value::CreateTensor<int64>(
            memory_info_handler, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size())));

        Ort::RunOptions runOptions;
        auto DecoderOutputTensors = DecoderSession->Run(runOptions, DecoderInputNames, inputTensors.data(),
                                                        inputTensors.size(), DecoderOutputNames, 1);

        auto &logits_output = DecoderOutputTensors[0];
        auto logits_ptr = logits_output.GetTensorMutableData<float>();
        auto logits_shape = logits_output.GetTensorTypeAndShapeInfo().GetShape();

        int logits_size = 1;
        for (size_t i = 0; i < logits_shape.size(); i++)
        {
            logits_size *= logits_shape[i];
        }
        // ALOGI("logits_size: %d", logits_size);
        logits.resize(logits_size);
        memcpy(logits.data(), logits_ptr, logits_size * sizeof(float));
    }
};
