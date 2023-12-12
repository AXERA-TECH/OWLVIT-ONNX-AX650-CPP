#include "Runner/OWLVIT_AX650.hpp"
#include "Runner/OWLVIT_Onnx.hpp"

#include "string_utility.hpp"
#include "cmdline.hpp"

struct Object
{
    std::string text;
    float prob;
    cv::Rect2f rect;
};

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

void get_out_bbox(std::vector<Object> &objects, int letterbox_rows, int letterbox_cols, int src_rows, int src_cols)
{
    /* yolov5 draw the result */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((letterbox_rows * 1.0 / src_rows) < (letterbox_cols * 1.0 / src_cols))
    {
        scale_letterbox = letterbox_rows * 1.0 / src_rows;
    }
    else
    {
        scale_letterbox = letterbox_cols * 1.0 / src_cols;
    }
    resize_cols = int(scale_letterbox * src_cols);
    resize_rows = int(scale_letterbox * src_rows);

    int tmp_h = (letterbox_rows - resize_rows) / 2;
    int tmp_w = (letterbox_cols - resize_cols) / 2;

    float ratio_x = (float)src_rows / resize_rows;
    float ratio_y = (float)src_cols / resize_cols;

    int count = objects.size();

    for (int i = 0; i < count; i++)
    {
        float x0 = (objects[i].rect.x);
        float y0 = (objects[i].rect.y);
        float x1 = (objects[i].rect.x + objects[i].rect.width);
        float y1 = (objects[i].rect.y + objects[i].rect.height);

        x0 = (x0 - tmp_w) * ratio_x;
        y0 = (y0 - tmp_h) * ratio_y;
        x1 = (x1 - tmp_w) * ratio_x;
        y1 = (y1 - tmp_h) * ratio_y;

        x0 = std::max(std::min(x0, (float)(src_cols - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(src_rows - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(src_cols - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(src_rows - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}

int main(int argc, char *argv[])
{
    std::string image_src;
    std::string text_src;
    std::string vocab_path;
    std::string image_encoder_model_path;
    std::string text_encoder_model_path;
    std::string decoder_model_path;
    float bbox_threshold = 0.2;
    int num_thread = 8;

    cmdline::parser cmd;
    cmd.add<std::string>("ienc", 0, "encoder model(onnx model or axmodel)", true, image_encoder_model_path);
    cmd.add<std::string>("tenc", 0, "text encoder model(onnx model or axmodel)", true, text_encoder_model_path);
    cmd.add<std::string>("dec", 'd', "decoder model(onnx)", true, decoder_model_path);
    cmd.add<std::string>("image", 'i', "image file (jpg png etc....)", true, image_src);
    cmd.add<std::string>("text", 't', "text or txt file", true, text_src);
    cmd.add<std::string>("vocab", 'v', "vocab path", true, vocab_path);
    cmd.add<float>("threshold", 0, "bbox threshold", false, bbox_threshold);
    cmd.add<int>("thread", 0, "thread num", false, num_thread);

    cmd.parse_check(argc, argv);

    vocab_path = cmd.get<std::string>("vocab");
    image_encoder_model_path = cmd.get<std::string>("ienc");
    text_encoder_model_path = cmd.get<std::string>("tenc");
    decoder_model_path = cmd.get<std::string>("dec");
    num_thread = cmd.get<int>("thread");

    std::shared_ptr<OWLVIT> mOWLVIT;
    if (string_utility<std::string>::ends_with(image_encoder_model_path, ".onnx"))
    {
        mOWLVIT.reset(new OWLVITOnnx);
    }
    else if (string_utility<std::string>::ends_with(image_encoder_model_path, ".axmodel"))
    {
        mOWLVIT.reset(new OWLVITAX650);
    }
    else
    {
        fprintf(stderr, "no impl for %s\n", image_encoder_model_path.c_str());
        return -1;
    }

    mOWLVIT->set_num_thread(num_thread);
    mOWLVIT->load_image_encoder(image_encoder_model_path);
    mOWLVIT->load_text_encoder(text_encoder_model_path);
    mOWLVIT->load_decoder(decoder_model_path);
    mOWLVIT->load_tokenizer(vocab_path);

    image_src = cmd.get<std::string>("image");
    text_src = cmd.get<std::string>("text");
    ALOGI("image_src [%s]", image_src.c_str());
    ALOGI("text_src [%s]", text_src.c_str());

    std::vector<std::string> texts;
    if (string_utility<std::string>::ends_with(text_src, ".txt"))
    {
        std::ifstream infile;
        infile.open(text_src);
        if (!infile.good())
        {
            ALOGE("");
            return -1;
        }

        std::string s;
        while (getline(infile, s))
        {
            texts.push_back(s);
        }
        infile.close();
    }
    else
    {
        texts.push_back(text_src);
    }
    std::vector<std::vector<float>> text_features;
    std::vector<std::vector<int64>> input_ids;
    auto time_start = std::chrono::high_resolution_clock::now();
    mOWLVIT->encode(texts, input_ids, text_features);
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = time_end - time_start;
    std::cout << "encode text Inference Cost time : " << diff.count() << "s" << std::endl;

    std::vector<float> image_features;
    std::vector<cv::Rect2f> pred_boxes;
    std::vector<std::string> image_paths;
    cv::Mat src = cv::imread(image_src);
    if (!src.data)
    {
        ALOGE("no image");
    }
    mOWLVIT->encode(src, image_features, pred_boxes);
    image_paths.push_back(image_src);

    float prob_threshold_u_sigmoid = -1.0f * (float)std::log((1.0f / bbox_threshold) - 1.0f);

    std::vector<Object> objects;

    time_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < input_ids.size(); i++)
    {
        std::vector<float> logits;
        mOWLVIT->decode(image_features, text_features[i], input_ids[i], logits);

        for (size_t j = 0; j < logits.size(); j++)
        {
            // float score = sigmoid(logits[j]);
            // printf("%f %f\n", score, logits[j]);
            if (logits[j] > prob_threshold_u_sigmoid)
            {
                Object obj;
                obj.text = texts[i];
                obj.prob = sigmoid(logits[j]);
                obj.rect = pred_boxes[j];
                objects.push_back(obj);
            }
        }
    }
    get_out_bbox(objects, 768, 768, src.rows, src.cols);
    time_end = std::chrono::high_resolution_clock::now();
    diff = time_end - time_start;
    std::cout << "post Inference Cost time : " << diff.count() << "s" << std::endl;

    for (size_t i = 0; i < objects.size(); i++)
    {
        printf("%s %f %f %f %f\n", objects[i].text.c_str(), objects[i].rect.x, objects[i].rect.y, objects[i].rect.width, objects[i].rect.height);
        cv::rectangle(src, objects[i].rect, cv::Scalar(0, 255, 0), 2);
        cv::putText(src, objects[i].text, cv::Point(objects[i].rect.x, objects[i].rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0));
    }
    cv::imwrite("result.jpg", src);

    return 0;
}