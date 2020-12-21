//
// Created by Bardakoff, Alexandre (IntlAssoc) on 12/21/20.
//

#ifndef HEDGEHOG_COLOR_SCHEME_H
#define HEDGEHOG_COLOR_SCHEME_H

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Enum color options
enum class ColorScheme {
  NONE, ///< No added coloration
  EXECUTION, ///< Colors nodes based on execution time
  WAIT ///< Colors nodes based on wait time
};
}

#endif //HEDGEHOG_COLOR_SCHEME_H
