//
// Created by Bardakoff, Alexandre (IntlAssoc) on 12/21/20.
//

#ifndef HEDGEHOG_DEBUG_OPTIONS_H
#define HEDGEHOG_DEBUG_OPTIONS_H

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Enum to enable debug printing
enum class DebugOptions {
  NONE, ///< No added debug options
  ALL ///< Shows debug information such as pointer addresses for nodes and edges
};
}

#endif //HEDGEHOG_DEBUG_OPTIONS_H
