#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "QLabel"
#include "QGridLayout"
#include "QFileDialog"
#include "myqlabel.h"

#include "owlvit/string_utility.hpp"

// #include "internal_func.hpp"


MainWindow::MainWindow(
    std::string vocab_path,
    std::string image_encoder_model_path,
    std::string text_encoder_model_path,
    std::string decoder_model_path,
    QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{

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
        return;
    }

    mOWLVIT->set_num_thread(8);
    mOWLVIT->load_image_encoder(image_encoder_model_path);
    mOWLVIT->load_text_encoder(text_encoder_model_path);
    mOWLVIT->load_decoder(decoder_model_path);
    mOWLVIT->load_tokenizer(vocab_path);

    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btn_search_clicked()
{
    if (ui->txt_context->text().isEmpty())
    {
        return;
    }

    std::vector<std::string> texts{ui->txt_context->text().toStdString()};

    auto time_start = std::chrono::high_resolution_clock::now();
    mOWLVIT->encode(texts, input_ids, text_features);
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = time_end - time_start;
    std::cout << "encode text Inference Cost time : " << diff.count() << "s" << std::endl;

    double threshold = 0.01;
    bool isOk = false;
    threshold = ui->txt_threshold->text().toDouble(&isOk);
    if (!isOk)
    {
        threshold = 0.01;
    }
    ALOGI("threshold = %f", threshold);

    float prob_threshold_u_sigmoid = -1.0f * (float)std::log((1.0f / threshold) - 1.0f);

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
    get_out_bbox(objects, 768, 768, src_rows, src_cols);
    time_end = std::chrono::high_resolution_clock::now();
    diff = time_end - time_start;
    std::cout << "post Inference Cost time : " << diff.count() << "s" << std::endl;
    ui->label->SetObjects(objects);
}

void MainWindow::on_txt_context_returnPressed()
{
    on_btn_search_clicked();
}

void MainWindow::on_btn_select_clicked()
{
    auto filename = QFileDialog::getOpenFileName(this, "", "", "image(*.png *.jpg *.jpeg *.bmp)");
    if (filename.isEmpty())
    {
        return;
    }

    cv::Mat src = cv::imread(filename.toStdString());

    if (src.data)
    {
        mOWLVIT->encode(src, image_features, pred_boxes);
        src_cols = src.cols;
        src_rows = src.rows;

        QImage img(src.data, src.cols, src.rows, src.cols * src.channels(), QImage::Format_BGR888);
        this->ui->label->SetImage(img);
    }
}
