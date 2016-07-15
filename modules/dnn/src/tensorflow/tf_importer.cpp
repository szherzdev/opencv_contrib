/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
using namespace cv;
using namespace cv::dnn;

#if HAVE_PROTOBUF
#include "graph.pb.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "tf_io.hpp"

using ::google::protobuf::RepeatedField;
using ::google::protobuf::RepeatedPtrField;
using ::google::protobuf::Message;
using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Reflection;

namespace
{

class TensorflowImporter : public Importer
{
    tensorflow::GraphDef net;

public:

    TensorflowImporter(const char *model)
    {
        if (model && model[0])
            ReadTFNetParamsFromBinaryFileOrDie(model, &net);
    }

    BlobShape blobShapeFromTensor(const tensorflow::TensorProto &tensor)
    {
        if (tensor.has_tensor_shape())
        {
            const tensorflow::TensorShapeProto &_shape = tensor.tensor_shape();
            BlobShape shape = BlobShape::all(_shape.dim_size());

            for (int i = 0; i < _shape.dim_size(); i++)
                shape[i] = (int)_shape.dim(i).size();

            return shape;
        }
        else
        {
            CV_Error(Error::StsError, "Unknown shape of input tensor");
            return BlobShape();
        }
    }

    // TODO: blobFromTensor for NCHW to NHWC reorder
    void blobFromTensor(const tensorflow::TensorProto &tensor, cv::dnn::Blob &dstBlob)
    {
        BlobShape shape = blobShapeFromTensor(tensor);

        // TODO: other blob types
        CV_Assert(tensor.dtype() == tensorflow::DT_FLOAT);

        if (shape.dims() == 4)
        {
            // REORDER blob NHWC to NCHW
            swap(shape[2], shape[3]); // NHCW
            swap(shape[1], shape[2]); // NCHW
        }

        dstBlob.create(shape, CV_32F);

        int size = tensor.tensor_content().size() / sizeof(float);
        CV_Assert(size == (int)dstBlob.matRefConst().total());

        float *dstData = dstBlob.matRef().ptr<float>();
        const float *data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());

        if (shape.dims() == 4)
        {
            // TODO: !!!! REORDER !!!!
            for (int i = 0; i < size; i++)
                dstData[i] = data[i];
        } else {
            for (int i = 0; i < size; i++)
                dstData[i] = data[i];
        }
    }

    void kernelFromTensor(const tensorflow::TensorProto &tensor, cv::dnn::Blob &dstBlob)
    {
        BlobShape shape = blobShapeFromTensor(tensor);

        // TODO: other blob types
        CV_Assert(tensor.dtype() == tensorflow::DT_FLOAT);
        CV_Assert(shape.dims() == 4);

        // REORDER kernel HWIO to OIHW
        swap(shape[0], shape[2]); // IWHO
        swap(shape[1], shape[3]); // IOHW
        swap(shape[0], shape[1]); // OIHW

        dstBlob.create(shape, CV_32F);

        int size = tensor.tensor_content().size() / sizeof(float);
        CV_Assert(size == (int)dstBlob.matRefConst().total());

        float *dstData = dstBlob.matRef().ptr<float>();
        const float *data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());

        // TODO: !!!! REORDER !!!!
        for (int i = 0; i < size; i++)
            dstData[i] = data[i];
    }

    //        struct BlobNote
    //        {
    //            BlobNote(const std::string &_name, int _layerId, int _outNum) :
    //                name(_name.c_str()), layerId(_layerId), outNum(_outNum) {}

    //            const char *name;
    //            int layerId, outNum;
    //        };

    struct Input
    {
        Input(const std::string &_name, int _outNum) :
            name(_name), outNum(_outNum) {}

        const std::string name;
        int outNum;
    };

    Input parseInputName(const std::string &input)
    {
        size_t delimiter_pos = input.find_first_of(":");
        std::string name = input;
        int out = 0;
        if (delimiter_pos != std::string::npos)
        {
            name = input.substr(0, delimiter_pos);
            std::istringstream(input.substr(delimiter_pos + 1)) >> out;
        }
        return Input(name, out);
    }

    void printTensorShape(const tensorflow::TensorShapeProto &shape)
    {
        std::cout << "[ ";
        for (int d = 0; d < shape.dim_size(); d++)
            std::cout << shape.dim(d).name() <<
                         ":" << shape.dim(d).size() << " ";
        std::cout << "]";
    }

    void printTensor(const tensorflow::TensorProto &tensor)
    {
        printTensorShape(tensor.tensor_shape());

        if (tensor.tensor_content().empty())
            return;

        switch (tensor.dtype())
        {
        case 1:  // float
            {
                const float *data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                int size = tensor.tensor_content().size() / sizeof(float);
                for (int i = 0; i < std::min(10, size); i++)
                    std::cout << " " << data[i];
                if (size > 10)
                    std::cout << " ... " << size - 10 << " more";
                break;
            }
        case 3:  // int32
            {
                const int *data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                int size = tensor.tensor_content().size() / sizeof(int);
                for (int i = 0; i < std::min(10, size); i++)
                    std::cout << " " << data[i];
                if (size > 10)
                    std::cout << " ... " << size - 10 << " more";
                break;
            }
        default:
            CV_Error(Error::StsError, "Tensor type is not supported");
            break;
        }
    }

    void printList(const tensorflow::AttrValue::ListValue &val)
    {
        std::cout << "(";
        for (int i = 0; i < val.i_size(); i++)
            std::cout << " " << val.i(i);
        std::cout << " )";
    }

    bool hasLayerAttr(const tensorflow::NodeDef &layer, const std::string &name)
    {
        google::protobuf::Map<std::string, tensorflow::AttrValue> attr = layer.attr();
        return attr.find(name) != attr.end();
    }

    const tensorflow::AttrValue& getLayerAttr(const tensorflow::NodeDef &layer, const std::string &name)
    {
        return layer.attr().at(name);
    }

    void printLayerAttr(const tensorflow::NodeDef &layer)
    {
        std::cout << std::endl << layer.name() << ":" << layer.op();
        for (int ii = 0; ii < layer.input_size(); ii++)
            std::cout << "(" << layer.input(ii) << ")";
        std::cout << std::endl;
        google::protobuf::Map<std::string, tensorflow::AttrValue> attr
                = layer.attr();
        for (google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator ai = attr.begin();
             ai != attr.end(); ++ai)
        {
            std::cout << ai->first << ":";
            if (ai->first == "dtype" || ai->first == "T")
                std::cout << ai->second.i();
            else if (ai->first == "padding")
                std::cout << ai->second.s();
            else if (ai->first == "transpose_a" || ai->first == "transpose_b")
                std::cout << ai->second.b();
            //            else if (ai->first == "shape")
            //              printTensorShape(ai->second.shape());
            else if (ai->first == "strides" || ai->first == "ksize")
                printList(ai->second.list());
            else
                printTensor(ai->second.tensor());
            std::cout << std::endl;
        }
    }

    void setInput(std::map<String, int> layer_id, Net dstNet, int id, const tensorflow::NodeDef &layer, int ii)
    {
        Input inp = parseInputName(layer.input(ii));
        if (layer_id.find(inp.name) == layer_id.end())
            CV_Error(Error::StsError, "Input layer not found: " + inp.name);
        dstNet.connect(layer_id.at(inp.name), inp.outNum, id, ii);
    }

    void setInputs(std::map<String, int> layer_id, int id, const tensorflow::NodeDef &layer, Net dstNet)
    {
        for (int ii = 0; ii < layer.input_size(); ii++)
            setInput(layer_id, dstNet, id, layer, ii);
    }

    void setStrides(LayerParams &layerParams, const tensorflow::NodeDef &layer)
    {
        if (hasLayerAttr(layer, "strides"))
        {
            const tensorflow::AttrValue& val = getLayerAttr(layer, "strides");
            if (val.list().i_size() != 4 ||
                val.list().i(0) != 1 || val.list().i(3) != 1)
                CV_Error(Error::StsError, "Unsupported strides");
            layerParams.set("stride_h", static_cast<int>(val.list().i(1)));
            layerParams.set("stride_w", static_cast<int>(val.list().i(2)));
        }
    }

    void setDims(LayerParams &layerParams, const tensorflow::TensorProto &tensor)
    {
        BlobShape shape = blobShapeFromTensor(tensor);

        // TODO: other blob types
        CV_Assert(tensor.dtype() == tensorflow::DT_INT32);
        CV_Assert(shape.dims() == 1);

        int size = tensor.tensor_content().size() / sizeof(int);
        const int *data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
        layerParams.set("dim", DictValue::arrayInt(data, size));
    }

    void setKSize(LayerParams &layerParams, const tensorflow::NodeDef &layer)
    {
        if (hasLayerAttr(layer, "ksize"))
        {
            const tensorflow::AttrValue& val = getLayerAttr(layer, "ksize");
            if (val.list().i_size() != 4 ||
                val.list().i(0) != 1 || val.list().i(3) != 1)
                CV_Error(Error::StsError, "Unsupported ksize");
            layerParams.set("kernel_h", static_cast<int>(val.list().i(1)));
            layerParams.set("kernel_w", static_cast<int>(val.list().i(2)));
        }
        else
        {
            layerParams.set("kernel_h", 1);
            layerParams.set("kernel_w", 1);
        }
    }

    void setPadding(LayerParams &layerParams, const tensorflow::NodeDef &layer)
    {
        if (hasLayerAttr(layer, "padding") &&
            getLayerAttr(layer, "padding").s() == "SAME")
        {
            layerParams.set("pad_mode", PaddingMode::SAME);
        }
        else
        {
            layerParams.set("pad_mode", PaddingMode::VALID);
        }
    }

    void populateNet(Net dstNet)
    {
        int layersSize = net.node_size();

        // find all Const layers for params
        std::map<String, int> value_id;
        for (int li = 0; li < layersSize; li++)
        {
            const tensorflow::NodeDef &layer = net.node(li);
            String name = layer.name();
            String type = layer.op();

            if (type != "Const")
                continue;  // only Const parameters are supported

            if (layer.attr().find("value") != layer.attr().end())
            {
                value_id.insert(std::make_pair(name, li));
            }
        }
//        for (std::map<String, int>::const_iterator it = value_id.begin();
//             it != value_id.end(); ++it)
//        {
//            std::cout << it->first << "=>";
//            printTensor(net.node(it->second).attr().at("value").tensor());
//            std::cout << std::endl;
//        }

        std::map<String, int> layer_id;
        for (int li = 0; li < layersSize; li++)
        {
            const tensorflow::NodeDef &layer = net.node(li);
            String name = layer.name();
            String type = layer.op();
            LayerParams layerParams;

            if (type == "Conv2D")
            {
                printLayerAttr(layer);

                Input kernel_inp = parseInputName(layer.input(1));
                if (value_id.find(kernel_inp.name) == value_id.end())
                    CV_Error(Error::StsError, "Const kernel input not found");
                if (kernel_inp.outNum != 0)
                    CV_Error(Error::StsError, "Unsupported kernel input");
                const tensorflow::TensorProto &kernel =
                        net.node(value_id.at(kernel_inp.name)).attr().at("value").tensor();
                std::cout << "Kernel:" << kernel_inp.name << "-" << kernel.dtype() << ":";
                printTensor(kernel);
                std::cout << std::endl;

                BlobShape kshape = blobShapeFromTensor(kernel);
                layerParams.set("kernel_h", kshape[0]);
                layerParams.set("kernel_w", kshape[1]);
                layerParams.set("num_output", kshape[3]);

                // kernel only
                layerParams.set("bias_term", false);
                layerParams.blobs.resize(1);
                kernelFromTensor(kernel, layerParams.blobs[0]);

                setStrides(layerParams, layer);
                setPadding(layerParams, layer);

                int id = dstNet.addLayer(name, "Convolution", layerParams);
                layer_id[name] = id;

                // one input only
                setInput(layer_id, dstNet, id, layer, 0);
            }
            else if (type == "BiasAdd")
            {
//                printLayerAttr(layer);

                Input bias_inp = parseInputName(layer.input(1));
                if (value_id.find(bias_inp.name) == value_id.end())
                    CV_Error(Error::StsError, "Const bias input not found");
                if (bias_inp.outNum != 0)
                    CV_Error(Error::StsError, "Unsupported bias input");
                const tensorflow::TensorProto &bias =
                        net.node(value_id.at(bias_inp.name)).attr().at("value").tensor();
//                std::cout << "Bias:" << bias_inp.name << "-" << bias.dtype() << ":";
//                printTensor(bias);
//                std::cout << std::endl;

                layerParams.blobs.resize(1);
                blobFromTensor(bias, layerParams.blobs[0]);

                int id = dstNet.addLayer(name, "Shift", layerParams);
                layer_id[name] = id;

                // one input only
                setInput(layer_id, dstNet, id, layer, 0);
            }
            else if (type == "MatMul")
            {
                printLayerAttr(layer);

                Input kernel_inp = parseInputName(layer.input(1));
                if (value_id.find(kernel_inp.name) == value_id.end())
                    CV_Error(Error::StsError, "Const kernel input not found");
                if (kernel_inp.outNum != 0)
                    CV_Error(Error::StsError, "Unsupported kernel input");
                const tensorflow::TensorProto &kernel =
                        net.node(value_id.at(kernel_inp.name)).attr().at("value").tensor();
                std::cout << "Kernel:" << kernel_inp.name << "-" << kernel.dtype() << ":";
                printTensor(kernel);
                std::cout << std::endl;

                BlobShape kshape = blobShapeFromTensor(kernel);
                layerParams.set("num_output", kshape[0]);

                // kernel only
                layerParams.set("bias_term", false);
                layerParams.blobs.resize(1);
                blobFromTensor(kernel, layerParams.blobs[0]);

                int id = dstNet.addLayer(name, "InnerProduct", layerParams);
                layer_id[name] = id;

                // one input only
                setInput(layer_id, dstNet, id, layer, 0);
            }
            else if (type == "Reshape")
            {
                printLayerAttr(layer);

                Input sizes_inp = parseInputName(layer.input(1));
                if (value_id.find(sizes_inp.name) == value_id.end())
                    CV_Error(Error::StsError, "Const kernel input not found");
                if (sizes_inp.outNum != 0)
                    CV_Error(Error::StsError, "Unsupported kernel input");
                const tensorflow::TensorProto &sizes =
                        net.node(value_id.at(sizes_inp.name)).attr().at("value").tensor();
                std::cout << "Sizes:" << sizes_inp.name << "-" << sizes.dtype() << ":";
                printTensor(sizes);
                std::cout << std::endl;

                setDims(layerParams, sizes);

                int id = dstNet.addLayer(name, "Reshape", layerParams);
                layer_id[name] = id;

                // one input only
                setInput(layer_id, dstNet, id, layer, 0);
            }
            else if (type == "Const")
            {
//                printLayerAttr(layer);
            }
            else if (type == "Softmax")
            {
//                printLayerAttr(layer);
                int id = dstNet.addLayer(name, "Softmax", layerParams);
                layer_id[name] = id;

                setInputs(layer_id, id, layer, dstNet);
            }
            else if (type == "LRN")
            {
//                printLayerAttr(layer);
                // TODO: params
                int id = dstNet.addLayer(name, "LRN", layerParams);
                layer_id[name] = id;

                setInputs(layer_id, id, layer, dstNet);
            }
            else if (type == "Concat")
            {
//                printLayerAttr(layer);
                // TODO: "concat_dim" input(0) to axis param
                int id = dstNet.addLayer(name, "Concat", layerParams);
                layer_id[name] = id;

                // input(0) is concat_dim
                for (int ii = 1; ii < layer.input_size(); ii++)
                {
                    Input inp = parseInputName(layer.input(ii));
                    if (layer_id.find(inp.name) == layer_id.end())
                        CV_Error(Error::StsError, "Input layer not found: " + inp.name);
                    dstNet.connect(layer_id.at(inp.name), inp.outNum, id, ii - 1);
                }
            }
            else if (type == "Relu")
            {
//                printLayerAttr(layer);
                int id = dstNet.addLayer(name, "ReLU", layerParams);
                layer_id[name] = id;

                setInputs(layer_id, id, layer, dstNet);
            }
            else if (type == "MaxPool")
            {
                printLayerAttr(layer);
                layerParams.set("pool", "max");

                setKSize(layerParams, layer);
                setStrides(layerParams, layer);
                setPadding(layerParams, layer);

                int id = dstNet.addLayer(name, "Pooling", layerParams);
                layer_id[name] = id;

                setInputs(layer_id, id, layer, dstNet);
            }
            else if (type == "AvgPool")
            {
                printLayerAttr(layer);
                layerParams.set("pool", "ave");

                setKSize(layerParams, layer);
                setStrides(layerParams, layer);
                setPadding(layerParams, layer);

                int id = dstNet.addLayer(name, "Pooling", layerParams);
                layer_id[name] = id;

                setInputs(layer_id, id, layer, dstNet);
            }
            else if (type == "Identity")
            {
//                printLayerAttr(layer);
                int id = dstNet.addLayer(name, "Identity", layerParams);
                layer_id[name] = id;

                setInputs(layer_id, id, layer, dstNet);
            }
            else if (type == "Placeholder")
            {
                printLayerAttr(layer);
                std::vector<String> netInputs(1);
                netInputs[0] = name;
                //                  for (int inNum = 0; inNum < net.input_size(); inNum++)
                //                  {
                //                    addedBlobs.push_back(BlobNote(net.input(inNum), 0, inNum));
                //                    netInputs[inNum] = net.input(inNum);
                //                  }
                layer_id[name] = 0;  // fictive layer id
                dstNet.setNetInputs(netInputs);
            }
            else
            {
                printLayerAttr(layer);
                //                  std::cerr << "Unknown layer type: " << type << std::endl;
                CV_Error(Error::StsError, "Unknown layer type");
            }
        }
    }

    ~TensorflowImporter()
    {

    }


};

}

Ptr<Importer> cv::dnn::createTensorflowImporter(const String &model)
{
    return Ptr<Importer>(new TensorflowImporter(model.c_str()));
}

#else //HAVE_PROTOBUF

Ptr<Importer> cv::dnn::createTensorflowImporter(const String&)
{
    CV_Error(cv::Error::StsNotImplemented, "libprotobuf required to import data from TensorFlow models");
    return Ptr<Importer>();
}

#endif //HAVE_PROTOBUF
