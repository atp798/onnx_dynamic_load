/* *
 * Copyright (c) 2020, ATP
 * All rights reserved.
 * MIT License
 */

#include <cstdlib>
#include <thread>
#include <time.h>

#include "onnx/common/ir_pb_converter.h"

#include "onnxruntime_api_wrap.h"

using namespace ONNX_NAMESPACE;
using namespace ONNX_NAMESPACE::inference;

void func(int id);

int main(int argc, char *argv[]) {
  srand(time(NULL));

  const int num = 4;
  func(num + 2);
  std::thread threads[num];
  for (int i = 0; i < num; ++i) {
    threads[i] = std::thread(func, i);
  }
  for (int i = 0; i < num; ++i) {
    threads[i].join();
  }
  func(num + 2);
}

void func(int id) {
  for (int u = 0; u < 10; u++) {

    // prepare input data
    Tensor data;
    std::vector<int64_t> sizes{1, 3, 2, 2, 1};
    data.sizes().swap(sizes);
    data.elem_type() = TensorProto_DataType_FLOAT;
    auto tensor_size = std::accumulate(data.sizes().begin(), data.sizes().end(),
                                       (int64_t)1, std::multiplies<int64_t>{});
    for (decltype(tensor_size) i = 0; i < tensor_size; ++i) {
      float neg = (static_cast<float>(rand() % 997) / 997.0) < 0.5 ? -1.0 : 1.0;
      data.floats().push_back((static_cast<float>(rand() % 997) / 997.0) * neg);
    }
    data.setName("node");

    // construct graph
    std::shared_ptr<Graph> graph = std::make_shared<Graph>();
    OpSetID new_opset_version("ai.onnx", 7);
    graph->opset_versions_mutable().emplace_back(std::move(new_opset_version));

    // construct graph input
    Value *input = graph->addInput();
    input->setUniqueName("node_in");
    input->setElemType(TensorProto_DataType_FLOAT);
    std::vector<Dimension> dim_sizes{data.sizes().cbegin(),
                                     data.sizes().cend()};
    input->setSizes(dim_sizes);

    // construct a node
    Node *node = graph->create(kSigmoid, graph->inputs(), 1);
    node->setName("node");
    Value *output = node->outputs()[0];
    output->setUniqueName("node_out");
    output->setElemType(TensorProto_DataType_FLOAT);

    graph->registerOutput(node->outputs()[0]);
    graph->appendNode(node);

    // prepare model infer parameters
    ModelProto mp;
    ExportModelProto(&mp, graph);

    std::vector<std::string> input_names(1, input->uniqueName());
    std::vector<std::string> output_names(1, output->uniqueName());

    std::vector<Tensor> input_tensors;
    input_tensors.push_back(data);
    std::vector<Tensor> output_tensors(1);

    auto api_wrapper = OnnxApiWrapper::GetInstance();
    api_wrapper->InferModel(mp, input_names, input_tensors, output_names,
                            &output_tensors);

    for (int i = 0; i < output_tensors[0].floats().size(); ++i) {
      printf("sigmoid of [%f] is: [%f]\n", data.floats()[i],
             output_tensors[0].floats()[i]);
      if (i > 1) {
        break;
      } // No need to print all results.
    }
    printf("Infer: %d times of thread: %d has finish.\n", u, id);
  }
}
