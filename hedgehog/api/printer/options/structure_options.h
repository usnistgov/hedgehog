//
// Created by Bardakoff, Alexandre (IntlAssoc) on 12/21/20.
//

#ifndef HEDGEHOG_STRUCTURE_OPTIONS_H
#define HEDGEHOG_STRUCTURE_OPTIONS_H

/// @brief Hedgehog main namespace
namespace hh {

/// @brief Enum structural options
enum class StructureOptions {
  NONE, ///< No added structural options
  ALLTHREADING, ///< Displays all tasks in clusters
  QUEUE, ///< Displays queue details (max queue size and queue size along edges)
  ALL ///< Displays both ALLTHREADING and QUEUE
};
}

#endif //HEDGEHOG_STRUCTURE_OPTIONS_H
