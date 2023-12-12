#include "mainwindow.h"

#include <QApplication>
#include "style/DarkStyle.h"

#include "owlvit/cmdline.hpp"

int main(int argc, char *argv[])
{
    std::string vocab_path;
    std::string image_encoder_model_path;
    std::string text_encoder_model_path;
    std::string decoder_model_path;

    cmdline::parser cmd;
    cmd.add<std::string>("ienc", 0, "encoder model(onnx model or axmodel)", true, image_encoder_model_path);
    cmd.add<std::string>("tenc", 0, "text encoder model(onnx model or axmodel)", true, text_encoder_model_path);
    cmd.add<std::string>("dec", 'd', "decoder model(onnx)", true, decoder_model_path);
    cmd.add<std::string>("vocab", 'v', "vocab path", true, vocab_path);

    cmd.parse_check(argc, argv);

    vocab_path = cmd.get<std::string>("vocab");
    image_encoder_model_path = cmd.get<std::string>("ienc");
    text_encoder_model_path = cmd.get<std::string>("tenc");
    decoder_model_path = cmd.get<std::string>("dec");

    QApplication a(argc, argv);
    QApplication::setStyle(new DarkStyle);
    MainWindow w(vocab_path,
                 image_encoder_model_path,
                 text_encoder_model_path,
                 decoder_model_path);
    w.show();
    return a.exec();
}
