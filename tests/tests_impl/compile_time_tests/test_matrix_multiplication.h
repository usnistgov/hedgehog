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


#ifdef HH_ENABLE_HH_CX

#include "../../data_structures/matrix_multiplication/data/matrix_data.h"
#include "../../data_structures/matrix_multiplication/data/matrix_block_data.h"
#include "../../data_structures/matrix_multiplication/task/addition_task.h"
#include "../../data_structures/matrix_multiplication/task/product_task.h"
#include "../../data_structures/matrix_multiplication/task/matrix_row_traversal_task.h"
#include "../../data_structures/matrix_multiplication/task/matrix_column_traversal_task.h"
#include "../../data_structures/matrix_multiplication/state/input_block_state.h"
#include "../../data_structures/matrix_multiplication/state/partial_computation_state.h"
#include "../../data_structures/matrix_multiplication/state/partial_computation_state_manager.h"

constexpr auto constructMatrixMultiplicationGraph() {
  using MatrixType = double;
  constexpr Order Ord = Order::Row;
  using GraphType = hh::Graph<
      3,
      MatrixData<MatrixType, 'a', Ord>, MatrixData<MatrixType, 'b', Ord>, MatrixData<MatrixType, 'c', Ord>,
      MatrixBlockData<MatrixType, 'c', Ord>
  >;
  hh_cx::Node<MatrixRowTraversalTask<MatrixType, 'a', Ord>> taskTraversalA("taskTraversalA");
  hh_cx::Node<MatrixColumnTraversalTask<MatrixType, 'b', Ord>> taskTraversalB("taskTraversalB");
  hh_cx::Node<MatrixRowTraversalTask<MatrixType, 'c', Ord>> taskTraversalC("taskTraversalC");
  hh_cx::Node<ProductTask<MatrixType, Ord>> productTask("productTask");
  hh_cx::Node<AdditionTask<MatrixType, Ord>> additionTask("additionTask");
  hh_cx::Node<hh::StateManager<
      2,
      MatrixBlockData<MatrixType, 'a', Ord>, MatrixBlockData<MatrixType, 'b', Ord>, // Block as Input
      std::pair<
          std::shared_ptr<MatrixBlockData<MatrixType, 'a', Ord>>,
          std::shared_ptr<MatrixBlockData<MatrixType, 'b', Ord>>> // Pair of block as output
  >> stateManagerInputBlock("stateManagerInputBlock");
  hh_cx::Node<PartialComputationStateManager<MatrixType, Ord>>
      stateManagerPartialComputation("stateManagerPartialComputation");

  hh_cx::Graph<GraphType> matrixMultiplicationGraph("Matrix Multiplication Graph");

  matrixMultiplicationGraph.inputs(taskTraversalA);
  matrixMultiplicationGraph.inputs(taskTraversalB);
  matrixMultiplicationGraph.inputs(taskTraversalC);
  matrixMultiplicationGraph.edges(taskTraversalA, stateManagerInputBlock);
  matrixMultiplicationGraph.edges(taskTraversalB, stateManagerInputBlock);
  matrixMultiplicationGraph.edges(taskTraversalC, stateManagerPartialComputation);
  matrixMultiplicationGraph.edges(stateManagerInputBlock, productTask);
  matrixMultiplicationGraph.edges(productTask, stateManagerPartialComputation);
  matrixMultiplicationGraph.edges(stateManagerPartialComputation, additionTask);
  matrixMultiplicationGraph.edges(additionTask, stateManagerPartialComputation);
  matrixMultiplicationGraph.outputs(stateManagerPartialComputation);

  auto cycleTest = hh_cx::CycleTest < GraphType > {};
  auto dataRaceTest = hh_cx::DataRaceTest < GraphType > {};
  matrixMultiplicationGraph.addTest(&cycleTest);
  matrixMultiplicationGraph.addTest(&dataRaceTest);

  return matrixMultiplicationGraph;
}

void testMatrixMultiplication (){
  using MatrixType = double;
  constexpr Order Ord = Order::Row;

  constexpr auto defroster = hh_cx::createDefroster<&constructMatrixMultiplicationGraph>();

  ASSERT_TRUE(defroster.isValid());

  if constexpr (defroster.isValid()) {
    // Args
    size_t
        n = 10,
        m = 11,
        p = 12,
        blockSize = 2,
        numberThreadProduct = 1,
        numberThreadAddition = 4;

    size_t
        nBlocks = 0,
        mBlocks = 0,
        pBlocks = 0;

    // Allocate matrices
    MatrixType
        *dataA = nullptr,
        *dataB = nullptr,
        *dataC = nullptr;

    MatrixType valA[] = {
        5.342076, 6.714045, 1.776813, 1.458503, 5.992301, 7.356877, 9.704967, 3.1713, 9.253088, 5.815256, 7.295659,
        8.703664, 6.393818, 3.759143, 9.032453, 6.283826, 8.364071, 3.772746, 4.602647, 9.041046, 5.395716, 7.639841,
        3.552889, 9.292402, 1.972255, 3.070164, 7.611839, 0.8225689, 0.655722, 1.00268, 7.757754, 9.797453, 4.361527,
        5.731369, 0.1051256, 0.5369907, 2.526278, 8.009276, 4.890554, 8.305291, 9.69977, 1.196107, 1.457837, 8.944946,
        7.133956, 3.078962, 5.330245, 9.897558, 3.343099, 3.451813, 9.247258, 3.66669, 7.352021, 1.827359, 0.341997,
        5.140028, 9.573338, 7.513398, 0.3157809, 0.7380033, 9.675292, 3.00344, 2.132629, 1.32904, 5.958378, 5.020756,
        5.209684, 0.3324007, 8.863868, 3.735634, 9.734334, 1.158682, 3.361639, 1.487494, 6.881399, 2.191344, 3.485955,
        9.807673, 8.741581, 7.367143, 8.220471, 0.5882183, 5.296038, 5.608577, 8.02636, 0.04442903, 4.05297, 2.439453,
        9.091303, 8.354532, 3.338108, 7.419773, 5.522026, 2.067048, 0.1320207, 7.171411, 1.197013, 5.171878, 8.57188,
        9.644483, 0.6290177, 2.520649, 0.229421, 4.872199, 9.630494, 6.347255, 9.032429, 1.398479, 1.498873, 7.313523};

    MatrixType valB[] = {
        4.139718, 4.529329, 0.336605, 1.810222, 3.801725, 5.996832, 0.9132839, 0.790903, 4.019073, 7.386784, 2.149351,
        2.673233,
        2.530323, 5.814741, 4.736624, 4.898511, 1.928291, 1.876575, 2.298093, 4.722692, 5.868142, 6.516873, 0.4794664,
        8.639146,
        1.667072, 3.473217, 7.784602, 2.252332, 3.109591, 6.902006, 7.758949, 5.302371, 8.646875, 2.062985, 5.719791,
        7.141562,
        7.290571, 7.332814, 7.52303, 4.697859, 7.572637, 2.292121, 9.809462, 6.672845, 9.352632, 9.469197, 0.3478637,
        2.041653,
        4.480771, 6.714272, 9.133393, 0.2843502, 7.850596, 6.221536, 5.367549, 4.488484, 8.302843, 1.181407, 4.738803,
        5.921759,
        7.641094, 5.104209, 5.446091, 3.383181, 0.8113772, 1.312401, 2.869725, 2.117002, 8.87523, 7.672208, 9.078923,
        0.134389,
        6.147253, 1.08537, 4.828039, 0.04782712, 9.07663, 0.4986638, 9.286674, 1.372805, 8.109163, 5.650368, 2.010407,
        8.943649,
        6.696804, 1.828454, 7.437124, 4.195498, 8.357969, 6.010148, 8.516271, 6.651993, 9.440146, 2.770389, 6.485387,
        9.877133,
        9.101093, 1.449146, 0.324026, 7.424997, 5.565105, 2.74751, 7.584763, 8.096213, 2.265774, 4.741994, 8.124542,
        1.173222,
        7.444313, 7.822786, 1.880702, 1.658278, 1.910391, 2.442339, 8.690981, 8.032517, 6.219939, 2.346627, 7.27709,
        9.358558,
        3.902084, 1.352133, 9.143088, 9.417981, 2.925327, 0.8842191, 1.267685, 4.557736, 5.607377, 4.014123, 6.375663,
        2.83386};

    MatrixType valC[] = {
        0.7156747, 3.301089, 2.079683, 0.4550757, 1.180297, 4.537336, 9.793406, 2.299961, 8.769625, 5.772725, 8.253111,
        2.528979,
        5.859247, 4.666943, 7.496532, 0.8897097, 0.7807057, 3.411999, 8.484864, 7.407236, 8.203436, 9.575473, 3.639715,
        1.427038,
        8.161156, 3.586594, 0.07355081, 4.761949, 7.081149, 0.5439291, 6.200231, 2.505953, 8.597561, 6.825964, 5.949542,
        2.263348,
        2.448699, 6.991482, 9.742039, 4.497139, 0.2665896, 9.877613, 7.18888, 2.217185, 3.120972, 7.636557, 1.062806,
        4.498003,
        0.7665204, 2.553463, 1.864237, 8.691397, 2.342086, 4.681396, 0.8665139, 8.235, 3.033779, 3.102452, 1.239418,
        4.186428,
        9.988487, 5.28991, 2.641439, 5.356626, 6.078251, 9.126141, 7.013451, 0.3522114, 3.525197, 3.586421, 5.524129,
        3.030474,
        5.147668, 8.887225, 9.618879, 2.217275, 1.527184, 2.685251, 0.8860531, 1.48113, 1.451793, 5.067225, 9.755991,
        4.888042,
        9.766856, 5.783398, 9.045323, 4.61435, 3.705479, 8.93455, 1.70136, 6.069844, 7.11103, 5.844402, 7.943416,
        6.254995,
        1.678382, 2.235484, 4.943851, 5.212324, 9.676045, 2.370671, 9.502612, 5.492717, 8.967208, 4.331273, 7.55006,
        3.906815,
        3.927854, 4.673819, 4.657913, 2.050636, 8.328597, 2.625712, 5.066805, 1.274055, 9.936222, 4.604211, 5.754633,
        7.340378};

    MatrixType valTruth[] = {
        373.347714, 246.2860426, 306.3613321, 241.2887982, 302.5582744, 181.6916432, 358.5771234, 289.1542425,
        420.3727681,
        320.4196596, 332.9014273, 331.0282143,
        428.5310758, 323.5919591, 376.2447595, 296.4695133, 338.0897964, 238.6438031, 406.0160191, 357.7589735,
        495.0215841, 400.6851459, 354.6339322, 325.5103045,
        283.749299, 254.7091565, 229.1764791, 199.6881809, 217.2944301, 163.8274644, 283.5843965, 285.0077739,
        311.4307703,
        225.8408919, 243.88308, 286.4983848,
        291.6656681, 184.6730294, 332.2913947, 183.9223011, 301.9672952, 187.3849024, 291.453469, 211.3413101,
        385.7995003,
        241.470354, 256.6979245, 288.2872592,
        323.739594, 226.0768498, 264.9972567, 184.4850873, 315.9583102, 189.6593959, 369.9356219, 257.8850352,
        386.3678925,
        313.948134, 216.7417456, 274.0081074,
        256.3424918, 229.4734371, 257.9174804, 189.030832, 162.0303641, 164.744316, 241.1119409, 215.1841254,
        352.9928521,
        260.8529551, 261.6225492, 278.9242524,
        245.2079957, 202.0648677, 270.7859107, 151.4385122, 250.8103636, 204.3322379, 286.9454843, 228.9564857,
        317.1004766, 194.6478956, 240.245779, 233.6326571,
        316.126743, 273.7155961, 323.8728985, 202.4541047, 285.0395134, 227.6684886, 345.9960739, 268.0041555,
        450.5385669,
        343.0665512, 237.8202322, 355.9179762,
        292.330667, 272.6605941, 333.5336398, 241.8213162, 274.6984948, 216.349312, 298.4547494, 286.6995403,
        408.7004825,
        298.2953713, 246.4925046, 318.4627277,
        298.6686805, 191.3661364, 297.1405103, 183.2597623, 266.9913459, 193.9387109, 259.6897702, 194.5178433,
        392.4962386, 267.3547733, 292.043259, 269.5481338};


    // Allocate and fill the matrices' data
    dataA = new MatrixType[n * m]();
    dataB = new MatrixType[m * p]();
    dataC = new MatrixType[n * p]();

    std::copy(std::begin(valA), std::end(valA), dataA);
    std::copy(std::begin(valB), std::end(valB), dataB);
    std::copy(std::begin(valC), std::end(valC), dataC);

    // Wrap them to convenient object representing the matrices
    auto matrixA = std::make_shared<MatrixData<MatrixType, 'a'>>(n, m, blockSize, dataA);
    auto matrixB = std::make_shared<MatrixData<MatrixType, 'b'>>(m, p, blockSize, dataB);
    auto matrixC = std::make_shared<MatrixData<MatrixType, 'c'>>(n, p, blockSize, dataC);

    nBlocks = static_cast<size_t>(std::ceil(n / blockSize) + (n % blockSize == 0 ? 0 : 1)),
    mBlocks = static_cast<size_t>(std::ceil(m / blockSize) + (m % blockSize == 0 ? 0 : 1)),
    pBlocks = static_cast<size_t>(std::ceil(p / blockSize) + (p % blockSize == 0 ? 0 : 1));

    // Tasks
    auto taskTraversalA =
        std::make_shared<MatrixRowTraversalTask<MatrixType, 'a', Ord>>();
    auto taskTraversalB =
        std::make_shared<MatrixColumnTraversalTask<MatrixType, 'b', Ord>>();
    auto taskTraversalC =
        std::make_shared<MatrixRowTraversalTask<MatrixType, 'c', Ord>>();
    auto productTask =
        std::make_shared<ProductTask<MatrixType, Ord>>(numberThreadProduct, p);
    auto additionTask =
        std::make_shared<AdditionTask<MatrixType, Ord>>(numberThreadAddition);

    // State
    auto stateInputBlock =
        std::make_shared<InputBlockState<MatrixType, Ord>>(nBlocks, mBlocks, pBlocks);
    auto statePartialComputation =
        std::make_shared<PartialComputationState<MatrixType, Ord>>(nBlocks, pBlocks, nBlocks * mBlocks * pBlocks + 1);
    // The + 1 is due to the last block of C produced triggering end of computation

    // StateManager
    auto stateManagerInputBlock =
        std::make_shared<hh::StateManager<
            2,
            MatrixBlockData<MatrixType, 'a', Ord>, MatrixBlockData<MatrixType, 'b', Ord>, // Block as Input
            std::pair<
                std::shared_ptr<MatrixBlockData<MatrixType, 'a', Ord>>,
                std::shared_ptr<MatrixBlockData<MatrixType, 'b', Ord>>> // Pair of block as output
        >>(stateInputBlock, "Input State Manager");
    auto stateManagerPartialComputation =
        std::make_shared<PartialComputationStateManager<MatrixType, Ord>>(statePartialComputation);

    auto matrixMultiplicationGraph = defroster.map(
        "taskTraversalA", taskTraversalA,
        "taskTraversalB", taskTraversalB,
        "taskTraversalC", taskTraversalC,
        "stateManagerInputBlock", stateManagerInputBlock,
        "stateManagerPartialComputation", stateManagerPartialComputation,
        "additionTask", additionTask,
        "productTask", productTask
    );

    // Execute the graph
    matrixMultiplicationGraph->executeGraph();

    // Push the matrices
    matrixMultiplicationGraph->pushData(matrixA);
    matrixMultiplicationGraph->pushData(matrixB);
    matrixMultiplicationGraph->pushData(matrixC);

    // Notify push done
    matrixMultiplicationGraph->finishPushingData();

    // Wait for the graph to terminate
    matrixMultiplicationGraph->waitForTermination();

    //Print the result matrix
    for (size_t rowC = 0; rowC < matrixC->matrixHeight(); ++rowC) {
      for (size_t colC = 0; colC < matrixC->matrixWidth(); ++colC) {
        ASSERT_NEAR(
            matrixC->matrixData()[rowC * matrixC->matrixWidth() + colC],
            valTruth[rowC * matrixC->matrixWidth() + colC],
            0.0001
        );
      }
    }

    // Deallocate the Matrices
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
  }
}

#endif //HH_ENABLE_HH_CX

