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

#ifndef HEDGEHOG_PRINTABLE_ABSTRACTION_H
#define HEDGEHOG_PRINTABLE_ABSTRACTION_H

#include "../../../api/printer/printer.h"
#include "../../../tools/print_options.h"

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Hedgehog core namespace
namespace core {

/// @brief Hedgehog abstraction namespace
namespace abstraction {

/// @brief PrintAbstraction, used to determine if the node should be visited by the printer and colored
class PrintableAbstraction {
  tool::PrintOptions printOptions_; ///< Print Options

 public:
  /// @brief Print options const accessor
  /// @return Const print options
  [[nodiscard]] tool::PrintOptions const &printOptions() const { return printOptions_; }

  /// @brief Print options accessor
  /// @return Print options
  tool::PrintOptions &printOptions() { return printOptions_; }

  /// @brief Visitor method, used by the printer to visit all nodes in the graph
  /// @param printer Printer (visitor pattern)
  virtual void visit(Printer *printer) = 0;

};
}
}
}

#endif //HEDGEHOG_PRINTABLE_ABSTRACTION_H
