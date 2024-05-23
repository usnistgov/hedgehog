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

#ifndef HEDGEHOG_PRINT_OPTIONS_H
#define HEDGEHOG_PRINT_OPTIONS_H

/// Hedgehog tool namespace
namespace hh {
/// Hedgehog tool namespace
namespace tool {

/// @brief Node print options
struct PrintOptions {
  /// @brief Simple color representation
  struct Color {
    uint8_t
        r_ = 0, ///< Red color value
        g_ = 0, ///< Green color value
        b_ = 0, ///< Blue color value
        a_ = 0; ///< Alpha color value
  };
  Color background_ = {0xff, 0xff, 0xff, 0xff}; ///< Background color
  Color font_ = {0, 0, 0, 0xff}; ///< Font color

  /// @brief Background color accessor
  /// @return Background color
  [[nodiscard]] Color background() const { return background_; }
  /// @brief Font color accessor
  /// @return Font color
  [[nodiscard]] Color font() const { return font_; }

  /// @brief Background color setter
  /// @param background Color to set
  void background(Color background) { background_ = background; }

  /// @brief Font color setter
  /// @param font Color to set
  void font(Color font) { font_ = font; }
};

} // hh
} // tools

#endif //HEDGEHOG_PRINT_OPTIONS_H
