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


#ifndef HEDGEHOG_SWITCH_RULE_H
#define HEDGEHOG_SWITCH_RULE_H

#include <cstdio>

/// @brief Hedgehog behavior namespace
namespace hh::behavior {
/// @brief Behavior definition for dispatching data to a Graph managed by an AbstractExecutionPipeline
/// @tparam GraphInput Input data type
template<class GraphInput>
class SwitchRule {
 public:
  /// @brief Switch rule to determine if data should be sent to the graph graphId managed by an
  /// AbstractExecutionPipeline
  /// @param data Data to be sent to Graph graphId
  /// @param graphId Graph Id
  /// @return True if the data should be sent to the Graph graphId, else False
  virtual bool sendToGraph(std::shared_ptr<GraphInput> &data, size_t const &graphId) = 0;
};
}

#endif //HEDGEHOG_SWITCH_RULE_H
