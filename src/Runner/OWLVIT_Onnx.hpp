#pragma once
#include "OWLVIT.hpp"
#include "BaseRunner.hpp"

class OWLVITOnnx : public OWLVIT
{
private:
    std::shared_ptr<BaseRunner> m_encoder;
    // cv::Mat input;

public:
    bool load_image_encoder(std::string encoder_path) override
    {
        m_encoder = CreateRunner(RT_OnnxRunner);
        BaseConfig config;
        config.nthread = num_thread;
        config.onnx_model = encoder_path;
        m_encoder->load(config);

        input_width = m_encoder->getInputShape(0)[3];
        input_height = m_encoder->getInputShape(0)[2];
        ALOGI("input size %d %d", input_height, input_width);
        mat_input = cv::Mat(input_height, input_width, CV_8UC3);

        LEN_IMAGE_FEATURE = 1;
        image_features_shape.clear();
        for (size_t i = 0; i < m_encoder->getOutputShape(0).size(); i++)
        {
            LEN_IMAGE_FEATURE *= m_encoder->getOutputShape(0)[i];
            image_features_shape.push_back(m_encoder->getOutputShape(0)[i]);
        }
        ALOGI("image feature len %d", LEN_IMAGE_FEATURE);

        CNT_PRED_BOXES = m_encoder->getOutputShape(1)[1];

        ALOGI("pred box cnt  %d", CNT_PRED_BOXES);

        return true;
    }

    void encode(cv::Mat image, std::vector<float> &image_features, std::vector<cv::Rect2f> &pred_boxes) override
    {
        if (!m_encoder.get())
        {
            ALOGE("encoder not init");
            return;
        }
        get_input_data_letterbox(image, mat_input, true);
        // cv::imwrite("letterbox.jpg", mat_input);
        // cv::resize(image, mat_input, cv::Size(input_width, input_height));
        // cv::cvtColor(mat_input, mat_input, cv::COLOR_BGR2RGB);

        float *inputPtr = (float *)m_encoder->getInputPtr(0);

        uchar *img_data = mat_input.data;

        int letterbox_cols = input_width;
        int letterbox_rows = input_height;
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < letterbox_rows; h++)
            {
                for (int w = 0; w < letterbox_cols; w++)
                {
                    int in_index = h * letterbox_cols * 3 + w * 3 + c;
                    int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                    inputPtr[out_index] = (float(img_data[in_index]) - _mean_val[c]) * _std_val[c];
                }
            }
        }

        auto ret = m_encoder->inference();

        image_features.resize(LEN_IMAGE_FEATURE);
        memcpy(image_features.data(), m_encoder->getOutputPtr(0), LEN_IMAGE_FEATURE * sizeof(float));

        float *output = (float *)m_encoder->getOutputPtr(1);
        pred_boxes.resize(CNT_PRED_BOXES);
        for (size_t i = 0; i < CNT_PRED_BOXES; i++)
        {
            float xc = output[i * 4 + 0] * input_width;
            float yc = output[i * 4 + 1] * input_height;
            pred_boxes[i].width = output[i * 4 + 2] * input_width;
            pred_boxes[i].height = output[i * 4 + 3] * input_height;
            pred_boxes[i].x = xc - pred_boxes[i].width / 2;
            pred_boxes[i].y = yc - pred_boxes[i].height / 2;
        }
    }
};
