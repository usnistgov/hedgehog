---
layout: "page"
title: Tutorial 2 - Hadamard Product with Managed State
---

# Content
- [Goal](#goal)
- [Computation](#computation)
- [Data structure](#data-structure)
- [State and State Manager](#state-and-state-manager)
- [Task](#computation-task)
- [Graph](#graph)
- [Conclusion](#conclusion)

----

# Goal
The second tutorial aims to reuse the Hadamard (element-wise) product of two matrices A and B from [tutorial 1]({{site.url}}/tutorials/tutorial1) to introduce the concept of *multiple inputs*, *state*, and, *state manager*. 

Here is the base API that will be presented in this tutorial: 
* Manage the state of the computation,
* Define a task with multiple inputs.

----

# Computation
The computation is decomposed as follow:
1. Decompose the matrices into blocks (inside the graph), 
2. Do the element-wise product of A and B, and, store the result into C.
Because the decomposition of A, B, and C takes place within the graph, it is required to manage the state of the computation to be able to build the correct triplet for blocks A, B, and C.

----

# Data structure
We will use the same data structures as [tutorial 1]({{site.url}}/tutorials/tutorial1): 
* MatrixData<T, Id, Order>: A matrix, 
* MatrixBlockData<T, Id, Order>: A matrix block, 
* TripleMatrixBlockData<T, Order>: The corresponding block from matrix A, matrix B and matrix C.

These data structures are specialized with the following elements:
* Type: The type of the matrix elements, 
* Id: The matrix identifier, a, b, or, c,
* Ord: The way the matrix is ordered, row based, or, column based. 

----

# State and State Manager
The Hedgehog *State Manager* is a type of node that has been designed to manage the local state of the computation. There is no built-in representation of the global state in the Hedgehog API. 

To manage the state of computation two C++ objects are needed:
1. An abstract state: That's the object that will represent the state itself. It will hold the data structures that are needed to manage the computation. 
2. A state manager: A Hedgehog node that will manage the state. 


To demonstrate the usage of *state* and *state manager*, the code of the *BlockState* will be explained.

----

# State
In this computation, we want to form a triplet of blocks from the the blocks of matrix A, matrix B, and matrix C. To achieve that we create a class *BlockState* as follow:

```cpp
class BlockState : public AbstractState<
    TripletMatrixBlockData<Type, Ord>,
    MatrixBlockData<Type, 'a', Ord>, MatrixBlockData<Type, 'b', Ord>, MatrixBlockData<Type, 'c', Ord>>
```

We note here:
1. The declaration follows the same ordering as the *AbstractTask*, output type first then input types, 
2. There are multiple input types: Nodes in Hedgehog can accept multiple inputs, such as the *graph*, *state* or *task*.

The multiple inputs imply:
1. The node can accept data from nodes that produce one of its input types.
2. The different types of inputs are received and handled independently from the others.
3. Multiple execute method will need to be defined, one for each input type.

```cpp
void execute(std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> ptr) override { /*[...]*/ }
void execute(std::shared_ptr<MatrixBlockData<Type, 'b', Ord>> ptr) override { /*[...]*/ }
void execute(std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> ptr) override { /*[...]*/ }
```

Contrary to the task, the state does not provide their results directly to another node, but their reults are transferred to the *state manager* that will manage it. Data is enqueued by the state by using the push method:
```cpp
void execute(std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> ptr) override {
  /*[...]*/
  this->push(triplet);
}
```

In our example, we can receive any block (in any order) of matrix A, B or C, and we will produce a triplet of these blocks only if they correspond with eachother: they represent the same part of the matrix. When corresponding blocks are not available, a block is stored to await for the matching block(s) to arrive. 

For each type of matrix, we define a temporary storage for these blocks with accessors that permits the storagei and receiving of these blocks. We use this storage to test if a triplet of blocks is available and remove the blocks from the storage when they are pushed to the next task. For the storage of blocks from matrix A we will have:

```cpp
private:
// The temporary storage data structure
std::vector<std::shared_ptr<MatrixBlockData<Type, 'a', Ord>>> gridMatrixA_ = {}; 

// The getter
std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> matrixA(size_t i, size_t j) {
  return gridMatrixA_[i * gridWidth_ + j];
}

// The setter
void matrixA(std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> blockA){
  gridMatrixA_[blockA->rowIdx() * gridWidth_ + blockA->colIdx()] = blockA;
}

// The method to remove it
void resetA(size_t i, size_t j){ gridMatrixA_[i * gridWidth_ + j] = nullptr; }
```

The computation will be almost the same for each of the input types:
```cpp
void execute(std::shared_ptr<MatrixBlockData<Type, 'a', Ord>> ptr) override {
  std::shared_ptr<MatrixBlockData<Type, 'b', Ord>> blockB = nullptr;
  std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> blockC = nullptr;

  // We get the position of the block inside the grid of blocks
  auto
    rowIdx = ptr->rowIdx(),
    colIdx = ptr->colIdx();

  // We get and test if the other corresponding blocks are available 
  if((blockB = matrixB(rowIdx, colIdx)) && (blockC = matrixC(rowIdx, colIdx))){
    // If they are we remove them from the data structure
    resetB(rowIdx, colIdx);
    resetC(rowIdx, colIdx);
    // We create the triplet
    auto triplet = std::make_shared<TripletMatrixBlockData<Type, Ord>>();
    triplet->a_ = ptr;
    triplet->b_ = blockB;
    triplet->c_ = blockC;
    // We transfer it to the state manager
    this->push(triplet);
  }else{
    // If not, we store the block
    matrixA(ptr);
  }
}
```

With this code we achieve a rendezvous for matrices A, B, and C, which is independant from the traversal. 

State is accessed synchronously by the *state manager* via locks that are built-in to the state. This ensures that there will be no race conditions when calling the execute method.

----

# State Manager
The *state manager*, is a type of node, that holds *state*. 

The *state manager*, fired with an input type will sequentially process the data in the following order:
1. lock the *state*, 
2. transfer the input data to the *state*, 
3. launch the correspondant *execute* method, 
4. gather the outputs from the *state*,
5. add the outputs to its output edge
5. unlock the *state*. 

To note, multiple *state managers* are allowed to hold the same *state* instance, so it's possible to share the *state* at different points in the graph. Because it's locked, the computation into a state should be as minimal and as fast as possible. 

In this tutorial, we Hedgehog's default state manager, that is created as follows: 
```cpp
// Declaring and instantiating the state and state manager
auto inputState = std::make_shared<BlockState<MatrixType, Ord>>(matrixA->numBlocksRows(), matrixA->numBlocksCols());
auto inputstateManager =
  std::make_shared<
      StateManager<
          TripletMatrixBlockData<MatrixType, Ord>,
          MatrixBlockData<MatrixType, 'a', Ord>,
          MatrixBlockData<MatrixType, 'b', Ord>,
          MatrixBlockData<MatrixType, 'c', Ord>>>("Block State Manager", inputState);
```

----

# Computation task
We reuse the same task as shown in [tutorial 1]({{site.url}}/tutorials/tutorial1).

The *MatrixRowTraversalTask* is used to produce the different blocks from each matrix, which are gathered and distributed for corresponding triplets by the *state manager* to the computation task.

----

# Graph
The graph from [tutorial 1]({{site.url}}/tutorials/tutorial1) is changed slightly to operate with receiving three types of *MatrixData* for A, B, and C.
  

Below is the graph construction, the connection to the input of the graph must match at least one of the task input(s) and the output of the graph must match the output of task output.
```cpp
// Set The hadamard task as the task that will be connected to the graph inputs
graphHadamard.input(taskTraversalA);
graphHadamard.input(taskTraversalB);
graphHadamard.input(taskTraversalC);

// Link the traversal tasks to the input state manager
graphHadamard.addEdge(taskTraversalA, inputstateManager);
graphHadamard.addEdge(taskTraversalB, inputstateManager);
graphHadamard.addEdge(taskTraversalC, inputstateManager);

// Link the input state manager to the hadamard product
graphHadamard.addEdge(inputstateManager, hadamardProduct);

// Set The hadamard task as the task that will be connected to the graph output
graphHadamard.output(hadamardProduct);
```

And here is the visualization:
![Tutoriel2Graph](img/Tutorial2HadamardProductWithstate.png "Tutorial 2 Hadamard Product With State visualization")

----

# Conclusion
In this tutorial, we have demonstrated:
* How to create state,  
* How to use the default state manager, 
* How to use multiple input types.
