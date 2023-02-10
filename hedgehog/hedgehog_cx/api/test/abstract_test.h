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

#ifndef HEDGEHOG_CX_ABSTRACT_TEST_H_
#define HEDGEHOG_CX_ABSTRACT_TEST_H_

#ifdef HH_ENABLE_HH_CX
#include <utility>
#include <string>

#include "../graph.h"

/// @brief Hedgehog compile-time namespace
namespace hh_cx {

/// @brief Abstraction for testing graph representation at compile-time
/// @tparam GraphType Type of dynamic graph
template<tool::HedgehogDynamicGraphForStaticAnalysis GraphType>
class AbstractTest {
 private:
  std::string testName_{}; ///< Test name
  bool isGraphValid_ = true; ///< Flag for graph validity against the test
  std::string errorMessage_{}; ///< Error message if graph not valid

 public:
  /// @brief Constructing a graph from its name
  /// @param testName Name of the name
  constexpr explicit AbstractTest(std::string const & testName) : testName_(testName) {}

  /// @brief Default destructor
  constexpr virtual ~AbstractTest() = default;

  /// @brief Accessor to the test name
  /// @return Test name (std::string const &)
  [[nodiscard]] constexpr std::string const &testName() const { return testName_; }

  /// @brief Accessor to the graph validity against the test
  /// @return True if the graph is valid against the test, else false
  [[nodiscard]] constexpr bool isGraphValid() const { return isGraphValid_; }

  /// @brief Accessor the the error message
  /// @return Error message (std::string const &)
  [[nodiscard]] constexpr std::string const & errorMessage() const { return errorMessage_; }

  /// @brief Test the graph
  /// @param graph Graph to test
  constexpr virtual void test(hh_cx::Graph<GraphType> const  * graph) = 0;

 protected:
  /// @brief Setter of graph validity
  /// @param isGraphValid Graph's validity
  constexpr void graphValid(bool isGraphValid) { isGraphValid_ = isGraphValid; }

  /// @brief Append a message (as string) to the overhaul error message
  /// @param message Message to append
  constexpr void appendErrorMessage(std::string const & message){ this->errorMessage_.append(message); }
};

}
#endif //HH_ENABLE_HH_CX
#endif //HEDGEHOG_CX_ABSTRACT_TEST_H_
