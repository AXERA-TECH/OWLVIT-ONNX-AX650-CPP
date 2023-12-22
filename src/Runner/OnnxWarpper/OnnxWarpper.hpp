#pragma once
#include "../BaseRunner.hpp"

#include "onnxruntime_cxx_api.h"
#include "thread"
#include "map"

struct OnnxTensor : BaseTensor
{
    ONNXTensorElementDataType internal_type;
    std::shared_ptr<char> internal_data;
};

static std::map<ONNXTensorElementDataType, TensorDataType> OnnxTypeToTensorType = {
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, TENSOR_DATA_TYPE_UNDEFINED},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, TENSOR_DATA_TYPE_FLOAT},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, TENSOR_DATA_TYPE_DOUBLE},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, TENSOR_DATA_TYPE_INT8},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, TENSOR_DATA_TYPE_INT16},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, TENSOR_DATA_TYPE_INT32},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, TENSOR_DATA_TYPE_INT64},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, TENSOR_DATA_TYPE_UINT8},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, TENSOR_DATA_TYPE_UINT16},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, TENSOR_DATA_TYPE_UINT32},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, TENSOR_DATA_TYPE_UINT64},
};

static std::map<TensorDataType, std::string> TensorTypeToStr = {
    {TENSOR_DATA_TYPE_FLOAT, "float"},
    {TENSOR_DATA_TYPE_DOUBLE, "double"},
    {TENSOR_DATA_TYPE_INT8, "int8"},
    {TENSOR_DATA_TYPE_INT16, "int16"},
    {TENSOR_DATA_TYPE_INT32, "int32"},
    {TENSOR_DATA_TYPE_INT64, "int64"},
    {TENSOR_DATA_TYPE_UINT8, "uint8"},
    {TENSOR_DATA_TYPE_UINT16, "uint16"},
    {TENSOR_DATA_TYPE_UINT32, "uint32"},
    {TENSOR_DATA_TYPE_UINT64, "uint64"},
    {TENSOR_DATA_TYPE_UNDEFINED, "undefined"},
};

class OnnxRunner : virtual public BaseRunner
{
    Ort::Env env;
    Ort::Session session{nullptr};

    std::vector<std::string> inputs_name;
    std::vector<const char *> inputs_name_cstr;
    std::vector<OnnxTensor> inputs_data;
    std::vector<Ort::Value> inputs_tensor;

    std::vector<std::string> outputs_name;
    std::vector<const char *> outputs_name_cstr;
    std::vector<OnnxTensor> outputs_data;
    std::vector<Ort::Value> outputs_tensor;

    Ort::AllocatorWithDefaultOptions allocator;

    void AddTensor(OrtMemoryInfo *memory_info, ONNXTensorElementDataType type, size_t cnt_elem, std::vector<int64_t> &shape,
                   std::vector<Ort::Value> &tensors, std::vector<OnnxTensor> &tensor_data)
    {
        std::shared_ptr<char> data;
        switch (type)
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        {
            data.reset(new char[cnt_elem * sizeof(float)], std::default_delete<char[]>());
            tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, (float *)data.get(), cnt_elem, shape.data(), shape.size()));
        }
        break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        {
            data.reset(new char[cnt_elem * sizeof(uint8_t)], std::default_delete<char[]>());
            tensors.push_back(Ort::Value::CreateTensor<uint8_t>(memory_info, (uint8_t *)data.get(), cnt_elem, shape.data(), shape.size()));
        }
        break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        {
            data.reset(new char[cnt_elem * sizeof(int8_t)], std::default_delete<char[]>());
            tensors.push_back(Ort::Value::CreateTensor<int8_t>(memory_info, (int8_t *)data.get(), cnt_elem, shape.data(), shape.size()));
        }
        break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        {
            data.reset(new char[cnt_elem * sizeof(uint16_t)], std::default_delete<char[]>());
            tensors.push_back(Ort::Value::CreateTensor<uint16_t>(memory_info, (uint16_t *)data.get(), cnt_elem, shape.data(), shape.size()));
        }
        break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        {
            data.reset(new char[cnt_elem * sizeof(int16_t)], std::default_delete<char[]>());
            tensors.push_back(Ort::Value::CreateTensor<int16_t>(memory_info, (int16_t *)data.get(), cnt_elem, shape.data(), shape.size()));
        }
        break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        {
            data.reset(new char[cnt_elem * sizeof(int32_t)], std::default_delete<char[]>());
            tensors.push_back(Ort::Value::CreateTensor<int32_t>(memory_info, (int32_t *)data.get(), cnt_elem, shape.data(), shape.size()));
        }
        break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        {
            data.reset(new char[cnt_elem * sizeof(int64_t)], std::default_delete<char[]>());
            tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, (int64_t *)data.get(), cnt_elem, shape.data(), shape.size()));
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        {
            data.reset(new char[cnt_elem * sizeof(double)], std::default_delete<char[]>());
            tensors.push_back(Ort::Value::CreateTensor<double>(memory_info, (double *)data.get(), cnt_elem, shape.data(), shape.size()));
        }
        break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        {
            data.reset(new char[cnt_elem * sizeof(uint32_t)], std::default_delete<char[]>());
            tensors.push_back(Ort::Value::CreateTensor<uint32_t>(memory_info, (uint32_t *)data.get(), cnt_elem, shape.data(), shape.size()));
        }
        break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        {
            data.reset(new char[cnt_elem * sizeof(uint64_t)], std::default_delete<char[]>());
            tensors.push_back(Ort::Value::CreateTensor<uint64_t>(memory_info, (uint64_t *)data.get(), cnt_elem, shape.data(), shape.size()));
        }
        break;
        default:
            printf("not support type: %d\n", type);
            break;
        }

        OnnxTensor tensor;
        tensor.internal_type = type;
        tensor.type = OnnxTypeToTensorType.at(type);
        tensor.cnt_elem = cnt_elem;
        tensor.internal_data = data;
        tensor.data = tensor.internal_data.get();
        tensor_data.push_back(tensor);
    }

public:
    int load(BaseConfig &config) override
    {
        Ort::SessionOptions session_options;
        if (config.nthread <= 0)
        {
            session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
            session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        }
        else
        {
            session_options.SetInterOpNumThreads(config.nthread);
            session_options.SetIntraOpNumThreads(config.nthread);
        }

        // session_options
        // TensorRT加速开启，CUDA加速开启
        // OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0); // tensorRT
        // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        session = Ort::Session{env, config.onnx_model.c_str(), session_options};
        printf("\ninputs: \n");
        for (size_t i = 0; i < session.GetInputCount(); i++)
        {
            auto input_name = std::string(session.GetInputNameAllocated(i, allocator).get());
            inputs_name.push_back(input_name);

            printf("%20s: ", input_name.c_str());
            auto input_shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

            if (input_shape.size() && input_shape[0] < 0)
            {
                input_shape[0] = 1;
            }

            for (size_t j = 0; j < input_shape.size(); j++)
            {
                printf("%ld", input_shape[j]);
                if (j < (input_shape.size() - 1))
                    printf(" x ");
            }

            size_t cnt_elem = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementCount();
            auto type = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
            AddTensor(memory_info, type, cnt_elem, input_shape, inputs_tensor, inputs_data);
            printf("[%s] \n", TensorTypeToStr.at((inputs_data.end() - 1)->type).c_str());
        }
        printf("output: \n");
        for (size_t i = 0; i < session.GetOutputCount(); i++)
        {
            auto output_name = std::string(session.GetOutputNameAllocated(i, allocator).get());
            outputs_name.push_back(output_name);

            printf("%20s: ", output_name.c_str());
            auto output_shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            if (output_shape.size() && output_shape[0] < 0)
            {
                output_shape[0] = 1;
            }

            for (size_t j = 0; j < output_shape.size(); j++)
            {
                printf("%ld", output_shape[j]);
                if (j < (output_shape.size() - 1))
                    printf(" x ");
            }

            int cnt_elem = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementCount();
            auto type = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
            AddTensor(memory_info, type, cnt_elem, output_shape, outputs_tensor, outputs_data);
            printf(" [%s] \n", TensorTypeToStr.at((outputs_data.end() - 1)->type).c_str());
        }

        inputs_name_cstr.resize(inputs_name.size());
        for (size_t i = 0; i < inputs_name.size(); i++)
        {
            inputs_name_cstr[i] = inputs_name[i].c_str();
        }

        outputs_name_cstr.resize(outputs_name.size());
        for (size_t i = 0; i < outputs_name.size(); i++)
        {
            outputs_name_cstr[i] = outputs_name[i].c_str();
        }

        return 0;
    }

    int inference() override
    {
        Ort::RunOptions run_options;
        session.Run(run_options,
                    inputs_name_cstr.data(), inputs_tensor.data(), inputs_tensor.size(),
                    outputs_name_cstr.data(), outputs_tensor.data(), outputs_tensor.size());
        return 0;
    }

    int getInputCount() override
    {
        return inputs_tensor.size();
    }

    std::vector<int64_t> getInputShape(int idx) override
    {
        return inputs_tensor[idx].GetTensorTypeAndShapeInfo().GetShape();
    }

    std::string getInputName(int idx) override
    {
        return std::string(session.GetInputNameAllocated(idx, allocator).get());
    }

    const BaseTensor *getInput(int idx) override
    {
        return &inputs_data[idx];
    }

    int getOutputCount() override
    {
        return outputs_tensor.size();
    }

    std::vector<int64_t> getOutputShape(int idx) override
    {
        return outputs_tensor[idx].GetTensorTypeAndShapeInfo().GetShape();
    }

    std::string getOutputName(int idx) override
    {
        return std::string(session.GetOutputNameAllocated(idx, allocator).get());
    }

    const BaseTensor *getOutput(int idx) override
    {
        return &outputs_data[idx];
    }
};
