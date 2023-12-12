#pragma once
#include "OWLVIT.hpp"
#include "ax_model_runner_ax650.hpp"

class OWLVITAX650 : public OWLVIT
{
private:
    std::shared_ptr<ax_runner_base> m_encoder;

public:
    bool load_image_encoder(std::string encoder_path) override
    {
        m_encoder.reset(new ax_runner_ax650);
        m_encoder->init(encoder_path.c_str());
        input_height = m_encoder->get_algo_height();
        input_width = m_encoder->get_algo_width();
        ALOGI("input size %d %d", input_height, input_width);
        mat_input = cv::Mat(input_height, input_width, CV_8UC3, m_encoder->get_input(0).pVirAddr);

        // LEN_IMAGE_FEATURE = m_encoder->get_output(0).nSize;
        auto shape = m_encoder->get_output(1).vShape;
        LEN_IMAGE_FEATURE = 1;
        image_features_shape.clear();
        for (size_t i = 0; i < shape.size(); i++)
        {
            LEN_IMAGE_FEATURE *= shape[i];
            image_features_shape.push_back(shape[i]);
        }
        ALOGI("image feature len %d", LEN_IMAGE_FEATURE);

        CNT_PRED_BOXES = m_encoder->get_output(0).vShape[1];
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
        // cv::resize(image, input, cv::Size(input_width, input_height));
        // cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        auto ret = m_encoder->inference();

        image_features.resize(LEN_IMAGE_FEATURE);
        memcpy(image_features.data(), m_encoder->get_output(1).pVirAddr, LEN_IMAGE_FEATURE * sizeof(float));

        float *output = (float *)m_encoder->get_output(0).pVirAddr;
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
