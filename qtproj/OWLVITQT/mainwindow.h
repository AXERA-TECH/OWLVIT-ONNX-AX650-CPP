#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "owlvit/Runner/OWLVIT_AX650.hpp"
#include "owlvit/Runner/OWLVIT_Onnx.hpp"

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(std::string vocab_path,
               std::string image_encoder_model_path,
               std::string text_encoder_model_path,
               std::string decoder_model_path,
               QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_btn_search_clicked();

    void on_txt_context_returnPressed();

    void on_btn_select_clicked();

private:
    Ui::MainWindow *ui;

    std::shared_ptr<OWLVIT> mOWLVIT;

    std::vector<std::vector<float>> text_features;
    std::vector<std::vector<int64>> input_ids;

    std::vector<float> image_features;
    std::vector<cv::Rect2f> pred_boxes;

    int src_cols = 0, src_rows = 0;
};
#endif // MAINWINDOW_H
