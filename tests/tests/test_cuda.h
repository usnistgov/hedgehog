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


#ifndef HEDGEHOG_TESTS_TEST_CUDA_H
#define HEDGEHOG_TESTS_TEST_CUDA_H

#include "../data_structures/cuda_tasks/cuda_product_task.h"
#include "../data_structures/cuda_tasks/cuda_copy_in_gpu.h"
#include "../data_structures/cuda_tasks/cuda_copy_out_gpu.h"
#include "../data_structures/tasks/matrix_row_traversal_task.h"
#include "../data_structures/tasks/matrix_column_traversal_task.h"
#include "../data_structures/tasks/addition_task.h"
#include "../data_structures/datas/matrix_block_data.h"
#include "../data_structures/states/cuda_input_block_state.h"
#include "../data_structures/states/output_state.h"
#include "..//data_structures/state_managers/partial_computation_state_manager.h"


void testCUDA() {
  using MatrixType = double;
  constexpr Order Ord = Order::Column;

  // Args
  const size_t
      n = 10,
      m = 10,
      p = 10,
      nBlocks = 4,
      mBlocks = 4,
      pBlocks = 4,
      blockSize = 3,
      numberThreadProduct = 1,
      numberThreadAddition = 1;

  // Allocate matrices
  std::array<MatrixType, n * m> dataA{};
  std::array<MatrixType, m * p> dataB{};
  std::array<MatrixType, n * p> dataC{};

  dataA.fill(1.);
  dataB.fill(2.);
  dataC.fill(3.);

  // Wrap them to convenient object representing the matrices
  auto matrixA = std::make_shared<MatrixData<MatrixType, 'a', Ord>>
      (n, m, blockSize, dataA.data());
  auto matrixB = std::make_shared<MatrixData<MatrixType, 'b', Ord>>
      (m, p, blockSize, dataB.data());
  auto matrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>
      (n, p, blockSize, dataC.data());

  // Graph
  auto matrixMultiplicationGraph =
      hh::Graph<MatrixBlockData<MatrixType, 'c', Ord>,
                MatrixData<MatrixType, 'a', Ord>, MatrixData<MatrixType, 'b', Ord>, MatrixData<MatrixType, 'c', Ord>>
          ("Matrix Multiplication Graph");

  // Cuda tasks
  // Tasks
  auto copyInATask = std::make_shared<CudaCopyInGpu<MatrixType, 'a'>>(pBlocks, blockSize, n);
  auto copyInBTask = std::make_shared<CudaCopyInGpu<MatrixType, 'b'>>(nBlocks, blockSize, m);
  auto productTask = std::make_shared<CudaProductTask<MatrixType>>(p, numberThreadProduct);
  auto copyOutTask = std::make_shared<CudaCopyOutGpu<MatrixType>>(blockSize);

  // MemoryManagers
  auto cudaMemoryManagerA =
      std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'a'>, size_t>>(nBlocks + 4, blockSize);
  auto cudaMemoryManagerB =
      std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'b'>, size_t>>(pBlocks + 4, blockSize);
  auto cudaMemoryManagerProduct =
      std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'p'>, size_t>>(8, blockSize);

  // Connect the memory manager
  productTask->connectMemoryManager(cudaMemoryManagerProduct);
  copyInATask->connectMemoryManager(cudaMemoryManagerA);
  copyInBTask->connectMemoryManager(cudaMemoryManagerB);

  // Tasks
  auto taskTraversalA = std::make_shared<MatrixColumnTraversalTask<MatrixType, 'a', Order::Column>>();
  auto taskTraversalB = std::make_shared<MatrixRowTraversalTask<MatrixType, 'b', Order::Column>>();
  auto taskTraversalC = std::make_shared<MatrixRowTraversalTask<MatrixType, 'c', Order::Column>>();
  auto additionTask = std::make_shared<AdditionTask<MatrixType, Ord>>(numberThreadAddition);

  // State
  auto stateInputBlock = std::make_shared<CudaInputBlockState<MatrixType>>(nBlocks, mBlocks, pBlocks);
  auto stateOutput = std::make_shared<OutputState<MatrixType, Ord>>(nBlocks, pBlocks, mBlocks);
  auto statePartialComputation =
      std::make_shared<PartialComputationState<MatrixType, Ord>>(nBlocks, pBlocks, nBlocks * mBlocks * pBlocks);

  // StateManager
  std::shared_ptr<hh::StateManager<
      std::pair<
          std::shared_ptr<CudaMatrixBlockData<MatrixType, 'a'>>,
          std::shared_ptr<CudaMatrixBlockData<MatrixType, 'b'>>>,
      CudaMatrixBlockData<MatrixType, 'a'>, CudaMatrixBlockData<MatrixType, 'b' >>> stateManagerInputBlock =
      std::make_shared<hh::StateManager<
          std::pair<
              std::shared_ptr<
                  CudaMatrixBlockData
                      <MatrixType,
                       'a'>>,
              std::shared_ptr<CudaMatrixBlockData
                                  <MatrixType,
                                   'b'>>>,
          CudaMatrixBlockData<MatrixType, 'a'>, CudaMatrixBlockData<MatrixType, 'b' >>
      >("Input State Manager", stateInputBlock);
  auto stateManagerPartialComputation =
      std::make_shared<PartialComputationStateManager<MatrixType, Ord>>
          (statePartialComputation);

  auto stateManagerOutputBlock =
      std::make_shared<hh::StateManager<
          MatrixBlockData<MatrixType, 'c', Ord>,
          MatrixBlockData<MatrixType, 'c', Ord >>>("Output State Manager", stateOutput);

  // Build the graph
  matrixMultiplicationGraph.input(taskTraversalA);
  matrixMultiplicationGraph.input(taskTraversalB);
  matrixMultiplicationGraph.input(taskTraversalC);

  // Copy the blocks to the device (NVIDIA GPU)
  matrixMultiplicationGraph.addEdge(taskTraversalA, copyInATask);
  matrixMultiplicationGraph.addEdge(taskTraversalB, copyInBTask);

  // Connect to the State manager to wait for compatible block of A and B
  matrixMultiplicationGraph.addEdge(copyInATask, stateManagerInputBlock);
  matrixMultiplicationGraph.addEdge(copyInBTask, stateManagerInputBlock);

  // Do the CUDA product task
  matrixMultiplicationGraph.addEdge(stateManagerInputBlock, productTask);

  // Copy out the temporary block to the CPU for accumulation after the product
  matrixMultiplicationGraph.addEdge(productTask, copyOutTask);
  matrixMultiplicationGraph.addEdge(copyOutTask, stateManagerPartialComputation);

  // Use the same graph for the accumulation
  matrixMultiplicationGraph.addEdge(taskTraversalC, stateManagerPartialComputation);
  matrixMultiplicationGraph.addEdge(stateManagerPartialComputation, additionTask);
  matrixMultiplicationGraph.addEdge(additionTask, stateManagerPartialComputation);
  matrixMultiplicationGraph.addEdge(additionTask, stateManagerOutputBlock);
  matrixMultiplicationGraph.output(stateManagerOutputBlock);

  // Execute the graph
  matrixMultiplicationGraph.executeGraph();

  // Push the matrices
  matrixMultiplicationGraph.pushData(matrixA);
  matrixMultiplicationGraph.pushData(matrixB);
  matrixMultiplicationGraph.pushData(matrixC);

  // Notify push done
  matrixMultiplicationGraph.finishPushingData();

  // Wait for the graph to terminate
  matrixMultiplicationGraph.waitForTermination();

  // Shutdown cuBLAS
  cublasShutdown();

  ASSERT_EQ(std::all_of(dataC.begin(), dataC.end(), [](MatrixType &val) { return val == 23; }), true);
}

#endif //HEDGEHOG_TESTS_TEST_CUDA_H
