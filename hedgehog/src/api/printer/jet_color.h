
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

#ifndef HEDGEHOG_JET_COLOR_H
#define HEDGEHOG_JET_COLOR_H

#include <sstream>
#include <iomanip>

#include "options/color_picker.h"

/// @brief Hedgehog main namespace
namespace hh {

/// Jet color range.
/// @brief Return a color from the jet color range
class JetColor : public ColorPicker {
 public:
  /// @brief Default constructor
  JetColor() = default;

  /// @brief Default destructor
  ~JetColor() override = default;

  /// @brief Get RGB value for a duration within a range for the jet color range
  /// @param value Value to get the RGB color
  /// @param min Min value in the range
  /// @param range Range of values
  /// @return String representing the RGB values
  std::string getRGBFromRange(std::chrono::nanoseconds const &value,
                              std::chrono::nanoseconds const &min,
                              std::chrono::nanoseconds const &range) override {
    double
        dVal = (double) value.count(), dMin = (double) min.count(), dRange = (double) range.count(),
        dR = 1, dG = 1, dB = 1;

    std::ostringstream oss;

    if (dVal < dMin + 0.25 * dRange) {
      dR = 0;
      dG = (4 * dVal - dMin) / dRange;
    } else if (dVal < (dMin + 0.5 * dRange)) {
      dR = 0;
      dB = 1 + 4 * (dMin + 0.25 * dRange - dVal) / dRange;
    } else if (dVal < (dMin + 0.75 * dRange)) {
      dR = 4 * (dVal - dMin - 0.5 * dRange) / dRange;
      dB = 0;
    } else {
      dG = 1 + 4 * (dMin + 0.75 * dRange - dVal) / dRange;
      dB = 0;
    }

    oss << "\"#"
        << std::setfill('0') << std::setw(2) << std::hex << (uint16_t) std::clamp(dR * 255, 0., 255.)
        << std::setfill('0') << std::setw(2) << std::hex << (uint16_t) std::clamp(dG * 255, 0., 255.)
        << std::setfill('0') << std::setw(2) << std::hex << (uint16_t) std::clamp(dB * 255, 0., 255.)
        << "\"";

    return oss.str();
  }
};

}
#endif //HEDGEHOG_JET_COLOR_H
