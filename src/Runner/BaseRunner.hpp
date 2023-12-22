#pragma once
#include "string"
#include "vector"
#include "memory"

enum RunnerType
{
    RT_UNKNOWN,
    RT_OnnxRunner,
    RT_OpenvinoRunner,
    RT_TensorrtRunner,
    RT_END,
};

enum TensorDataType
{
    TENSOR_DATA_TYPE_UNDEFINED,

    TENSOR_DATA_TYPE_FLOAT,  // maps to c type float
    TENSOR_DATA_TYPE_DOUBLE, // maps to c type double

    TENSOR_DATA_TYPE_UINT8, // maps to c type uint8_t
    TENSOR_DATA_TYPE_INT8,  // maps to c type int8_t

    TENSOR_DATA_TYPE_UINT16, // maps to c type uint16_t
    TENSOR_DATA_TYPE_INT16,  // maps to c type int16_t

    TENSOR_DATA_TYPE_INT32,  // maps to c type int32_t
    TENSOR_DATA_TYPE_UINT32, // maps to c type uint32_t

    TENSOR_DATA_TYPE_INT64,  // maps to c type int64_t
    TENSOR_DATA_TYPE_UINT64, // maps to c type uint64_t
};

struct BaseConfig
{
    std::string onnx_model;
    std::string output_model; // for trt
    int nthread = 8;
};

struct BaseTensor
{
    TensorDataType type = TENSOR_DATA_TYPE_UNDEFINED;
    size_t cnt_elem;
    void *data = nullptr;
};

class BaseRunner
{
public:
    virtual int load(BaseConfig &config) = 0;
    virtual int inference() = 0;

    virtual int getInputCount() = 0;
    virtual std::vector<int64_t> getInputShape(int idx) = 0;
    virtual std::string getInputName(int idx) = 0;
    virtual const BaseTensor *getInput(int idx) = 0;

    virtual int getOutputCount() = 0;
    virtual std::vector<int64_t> getOutputShape(int idx) = 0;
    virtual std::string getOutputName(int idx) = 0;
    virtual const BaseTensor *getOutput(int idx) = 0;
};

std::shared_ptr<BaseRunner> CreateRunner(RunnerType rt);
