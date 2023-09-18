#ifndef HEDGEHOG_HEDGEHOG_SRC_API_PRINTER_OPTIONS_INPUT_OPTION_H_
#define HEDGEHOG_HEDGEHOG_SRC_API_PRINTER_OPTIONS_INPUT_OPTION_H_

/// @brief Hedgehog main namespace
namespace hh {
/// @brief Enum to choose the execution time format depending on the input types
enum class InputOptions {
  GATHERED, ///< Present the execution time for all  input type gathered
  SEPARATED ///< Shows the execution time per input type
};
}

#endif //HEDGEHOG_HEDGEHOG_SRC_API_PRINTER_OPTIONS_INPUT_OPTION_H_
