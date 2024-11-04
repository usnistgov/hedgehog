// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.

#ifndef HEDGEHOG_HEDGEHOG_EXPORT_FILE_H_
#define HEDGEHOG_HEDGEHOG_EXPORT_FILE_H_

#include <map>
#include <list>
#include <vector>
#include <fstream>
#include <unordered_set>

#include "printer.h"
#include "../../core/abstractions/base/node/graph_node_abstraction.h"
#include "../../core/abstractions/base/node/task_node_abstraction.h"
#include "../../core/abstractions/base/any_groupable_abstraction.h"

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Printer to produce a representation of the the graph structure in a json format for the GUI
class HedgehogExportFile : public Printer {
 private:
  std::ofstream outputFile_ = {}; ///< Output file stream
  std::ostringstream buffer_; ///< Buffer for the file

  std::map<std::pair<core::abstraction::NodeAbstraction const *, core::abstraction::NodeAbstraction const *>,
           std::list<std::string>> edges_; ///< list of edges to print
  std::map<core::abstraction::NodeAbstraction const *, std::pair<std::string, size_t>> mapExecutionPipelineAlias_; ///< Mapping of execution pipeline node and its switch to an alias name
 public:
  /// @brief Create a HedgehogExportFile from the graph pointer
  /// @details the graph's name is the name of the file + ".hhrojson"
  /// @param graph Pointer to the graph
  explicit HedgehogExportFile(core::abstraction::GraphNodeAbstraction const *graph) {
    outputFile_.open(graph->name() + ".hhrojson");
  }

  /// @brief Put the buffer content in the file and close the file
  ~HedgehogExportFile() override {
    outputFile_ << buffer_.str();
    outputFile_.close();
  }

  /// @brief Print graph header under json format
  /// @param graph Graph to print
  void printGraphHeader(core::abstraction::GraphNodeAbstraction const *graph) override {
    buffer_ << "{\n";
    // If the graph is the outer graph, i.e. the main graph
    if (!graph->isRegistered()) {
      buffer_ << "\"type\": \"graph\",\n";
    } else {
      buffer_ << "\"type\": \"innerGraph\",\n";
      buffer_ << R"("graphIdentifier": ")" << graph->graphId() << "\",\n";
    }
    buffer_ << R"("instanceIdentifier": ")" << graph->name() << "\",\n";
    // Open the node definitions
    buffer_ << R"("nodes": [)" << "\n";
  }

  /// @brief Print graph footer under json format
  /// @param graph Graph to print
  void printGraphFooter(core::abstraction::GraphNodeAbstraction const *graph) override {
    std::pair<std::string, size_t> infoSrc, infoDest;
    // Remove ",\n"
    buffer_.seekp(-2, buffer_.cur);
    // Close the node definitions
    buffer_ << "\n],\n";
    // If the graph is the outer graph, i.e. the main graph
    if (!graph->isRegistered()) {
      // Export edges
      buffer_ << "\"edges\": [\n";
      for (auto const &edge : edges_) {
        auto const &[src, dest] = edge.first;

        if(mapExecutionPipelineAlias_.contains(src)){ infoSrc = mapExecutionPipelineAlias_.at(src);}
        else { infoSrc = {src->name(), src->graphId()}; }

        if(mapExecutionPipelineAlias_.contains(dest)){ infoDest = mapExecutionPipelineAlias_.at(dest);}
        else { infoDest = {dest->name(), dest->graphId()}; }

        buffer_ << "{\n";
        buffer_
            << R"("sourceId": ")" << infoSrc.first << "\",\n"
            << R"("sourceGraphId": ")" << infoSrc.second << "\",\n"
            << R"("destId": ")" << infoDest.first << "\",\n"
            << R"("destGraphId": ")" << infoDest.second << "\",\n"
            << R"("types": [)" << "\n";
        auto const &types = edge.second;
        for (auto const &type : types) {
          buffer_ << "\"" << type << "\",\n";
        }
        // Remove ",\n"
        buffer_.seekp(-2, buffer_.cur);
        buffer_ << "\n]\n},\n";
      } // for all edges
      // Remove ",\n"
      buffer_.seekp(-2, buffer_.cur);
      buffer_ << "\n]\n}";
    } else { // Not main graph
      // Remove ",\n"
      buffer_.seekp(-2, buffer_.cur);
      buffer_ << "},\n";
    }// if else main graph
  }

  /// @brief Print execution pipeline header under the json format
  /// @param ep Execution pipeline to print
  /// @param switchNode Execution pipeline switch
  void printExecutionPipelineHeader(core::abstraction::ExecutionPipelineNodeAbstraction const *ep,
                                    [[maybe_unused]] core::abstraction::NodeAbstraction const *switchNode) override {
    buffer_ << "{\n";
    buffer_ << "\"type\": \"executionPipeline\",\n";
    buffer_ << R"("instanceIdentifier": ")" << ep->name() << "\",\n";
    buffer_ << R"("graphIdentifier": ")" << switchNode->graphId() << "\",\n";
    buffer_ << R"("switch": ")" << switchNode->name() << "\",\n";
    buffer_ << "\"graphs\": [";

    std::ostringstream oss;
    oss << ep->name() << " switch";

    mapExecutionPipelineAlias_.insert({ep, {oss.str(), switchNode->graphId()}});
    mapExecutionPipelineAlias_.insert({switchNode, {oss.str(), switchNode->graphId()}});
  }

  /// @brief Print execution pipeline footer under the json format
  void printExecutionPipelineFooter() override {
    // Remove ",\n"
    buffer_.seekp(-2, buffer_.cur);
    buffer_ << "\n]\n},";
  }

  /// @brief Print node information depending on the kind of node under the json format
  /// @param node Node to print
  void printNodeInformation(core::abstraction::NodeAbstraction *node) override {
    auto const &nodeAsTask = dynamic_cast<core::abstraction::TaskNodeAbstraction *>(node);
    auto const &nodeAsSM = dynamic_cast<core::abstraction::StateManagerNodeAbstraction const *>(node);
    auto copyableNode = dynamic_cast<core::abstraction::AnyGroupableAbstraction const *>(node);
    buffer_ << "{\n";
    if (nodeAsSM) {
      buffer_ << "\"type\": \"stateManager\",\n";
    } else if (nodeAsTask) {
      buffer_ << "\"type\": \"task\",\n";
    } else {
      buffer_ << "\"type\": \"other\",\n";
    }
    buffer_ << R"("instanceIdentifier": ")" << node->name() << "\",\n";
    buffer_ << R"("graphIdentifier": ")" << node->graphId() << "\",\n";
    if (nodeAsTask && nodeAsTask->hasMemoryManagerAttached()) {
      buffer_ << R"("managedMemory": ")" << nodeAsTask->memoryManager()->managedType() << "\",\n";
    }
    if (copyableNode) { buffer_ << R"("thread": )" << copyableNode->numberThreads() << "\n"; }
    else { buffer_ << R"("thread": 1)" << "\n"; }
    buffer_ << "},\n";
  }

  /// @brief Print an edge between node from and to for the type edgetype under the json format
  /// @param from Sender node
  /// @param to Receiving node
  /// @param edgeType Type of data transmitted through the edge
  /// @param queueSize Number of elements in the receiving queue [unused]
  /// @param maxQueueSize Maximum size of the receiving queue [unused]
  void printEdge(core::abstraction::NodeAbstraction const *from,
                 core::abstraction::NodeAbstraction const *to,
                 std::string const &edgeType,
                 [[maybe_unused]]size_t const &queueSize,
                 [[maybe_unused]]size_t const &maxQueueSize) override {
    std::pair<core::abstraction::NodeAbstraction const *, core::abstraction::NodeAbstraction const *> key{from, to};
    if (!edges_.contains(key)) { edges_.insert({key, {}}); }
    edges_[key].push_back(edgeType);
  }

  /// @brief Print a group of nodes under the json format
  /// @param representative Group node representative (only one actually printed)
  /// @param group Group of nodes [unused]
  /// @throw std::runtime_error if the group representative node does not derives from AnyGroupableAbstraction
  void printGroup(core::abstraction::NodeAbstraction *representative,
                  [[maybe_unused]]std::vector<core::abstraction::NodeAbstraction *> const &group) override {
    auto printRepr = dynamic_cast<core::abstraction::PrintableAbstraction *>(representative);
    if (printRepr == nullptr) {
      std::ostringstream oss;
      oss << "Internal error in: " << __FUNCTION__
          << " a group of node should be created with node that derives from AnyGroupableAbstraction and PrintableAbstraction";
      throw std::runtime_error(oss.str());
    }
    printRepr->visit(this);
  }

  /// @brief Do nothing, the source already exists by default in the GUI
  /// @param source Source to print (unused)
  void printSource([[maybe_unused]]core::abstraction::NodeAbstraction const *source) override {
    buffer_ << "{\n";
    buffer_ << "\"type\": \"IONode\",\n";
    buffer_ << R"("instanceIdentifier": ")" << source->name() << "\",\n";
    buffer_ << R"("graphIdentifier": ")" << source->graphId() << "\",\n";
    buffer_ << R"("thread": 1)" << "\n";
    buffer_ << "},\n";
  }
  /// @brief Do nothing, the sink already exists by default in the GUI
  /// @param sink Sink to print (unused)
  void printSink([[maybe_unused]]core::abstraction::NodeAbstraction const *sink) override {
    buffer_ << "{\n";
    buffer_ << "\"type\": \"IONode\",\n";
    buffer_ << R"("instanceIdentifier": ")" << sink->name() << "\",\n";
    buffer_ << R"("graphIdentifier": ")" << sink->graphId() << "\",\n";
    buffer_ << R"("thread": 1)" << "\n";
    buffer_ << "},\n";
  }
};
}
#endif //HEDGEHOG_HEDGEHOG_EXPORT_FILE_H_
