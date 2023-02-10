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
//
// Created by Bardakoff, Alexandre (IntlAssoc) on 12/4/20.
//

#ifndef HEDGEHOG_TEST_CRITICAL_PATH_H
#define HEDGEHOG_TEST_CRITICAL_PATH_H

#include "../../../../hedgehog/hedgehog.h"

#ifdef HH_ENABLE_HH_CX

template<class GraphType>
class TestCriticalPath : public hh_cx::AbstractTest<GraphType> {
 private:
  double_t
      maxPathValue_ = 0,
      currentPathValue_ = 0;

  hh_cx::PropertyMap<double>
      propertyMap_;

  hh_cx::Graph<GraphType> const
      *graph_ = nullptr;

  std::vector<hh_cx::behavior::AbstractNode const *>
      criticalVector_{},
      visited_{};

 public:
  constexpr explicit TestCriticalPath(hh_cx::PropertyMap<double> propertyMap)
      : hh_cx::AbstractTest<GraphType>("Critical Path"), propertyMap_(std::move(propertyMap)) {}

  constexpr ~TestCriticalPath() override = default;

  constexpr void test(hh_cx::Graph<GraphType> const *graph) override {
    graph_ = graph;

    auto const &inputNodeMap = graph->inputNodes();
    for (auto const &type : inputNodeMap.types()) {
      for (auto const &inputNode : inputNodeMap.nodes(type)) {
        this->visitNode(inputNode);
      }
    }

    if (criticalVector_.empty()) {
      this->graphValid(true);
    } else {
      this->graphValid(false);

      this->appendErrorMessage("The critical path is:\n\t");
      this->appendErrorMessage(criticalVector_.front()->name());

      for (size_t criticalNodeId = 1; criticalNodeId < criticalVector_.size(); ++criticalNodeId) {
        this->appendErrorMessage(" -> ");
        this->appendErrorMessage(criticalVector_.at(criticalNodeId)->name());
      }
    };
  }

 private:
  constexpr void visitNode(hh_cx::behavior::AbstractNode const *node) {
    if (std::find(visited_.cbegin(), visited_.cend(), node) == visited_.cend()) {
      currentPathValue_ += propertyMap_.property(node->name());
      visited_.push_back(node);

      if (std::find(visited_.cbegin(), visited_.cend(), node) != visited_.cend()) {
        if (currentPathValue_ > maxPathValue_) {
          maxPathValue_ = currentPathValue_;
          criticalVector_.clear();
          criticalVector_ = visited_;
        }
      }

      for (auto const &neighbor : graph_->adjacentNodes(node)) { visitNode(neighbor); }

      currentPathValue_ -= propertyMap_.property(node->name());
      visited_.pop_back();
    }
  }
};

#endif //HH_ENABLE_HH_CX

#endif //HEDGEHOG_TEST_CRITICAL_PATH_H
