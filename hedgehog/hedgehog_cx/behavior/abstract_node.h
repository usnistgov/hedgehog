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

#ifndef HEDGEHOG_CX_ABSTRACT_NODE_H_
#define HEDGEHOG_CX_ABSTRACT_NODE_H_

#ifdef HH_ENABLE_HH_CX

#include <string>
#include <map>
#include <utility>
#include <vector>

/// @brief Hedgehog compile-time namespace
namespace hh_cx {
/// @brief Hedgehog behavior namespace
namespace behavior {

/// @brief Node abstraction. Used to get hedgehog uniform type to store all Node in the Graph for the compile-time
/// analysis.
/// @attention It should NOT be used or derived by an end user. Use Node instead.
class AbstractNode {
 private:
  std::string name_{}; ///< Name of the node
 public:
  /// @brief AbstractNode constructor
  /// @param name Node's name
  constexpr explicit AbstractNode(std::string const & name) : name_(name) {}

  /// @brief Default destructor
  constexpr virtual ~AbstractNode() = default;

  /// @brief Name accessor
  /// @return Node's name
  [[nodiscard]] constexpr std::string const &name() const { return name_; }

  /// @brief Test if the canTerminate method has been overloaded in the dynamic node type
  /// @return True, if canTerminate method has been overloaded in the dynamic node type, else False
  [[nodiscard]] constexpr virtual bool isCanTerminateOverloaded() const { return false; }

  /// @brief Test if a type (represented by its name) is an input type declared read-only
  /// @param typeName Name of the type
  /// @return True if the type has been declared as read only, else false
  [[nodiscard]] constexpr virtual bool isTypeAnROType(std::string const &typeName) const = 0;

  /// @brief Test if a type (represented by its name) is a const input type
  /// @param typeName Name of the type
  /// @return True if the type is a const input type, else false
  [[nodiscard]] constexpr virtual bool isTypeAConstType(std::string const &typeName) const = 0;
};
}
}
#endif //HH_ENABLE_HH_CX
#endif //HEDGEHOG_CX_ABSTRACT_NODE_H_
