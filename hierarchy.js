var hierarchy =
[
    [ "hh::traits::_is_included_< T1, T2, Is >", "d8/d38/a00926.html", null ],
    [ "hh::traits::_is_included_< T1, T2, std::integer_sequence< std::size_t, Is... > >", "d7/d16/a00930.html", null ],
    [ "hh::traits::_is_included_< T1, T2, std::make_integer_sequence< std::size_t, std::tuple_size< T1 >::value > >", "d8/d38/a00926.html", [
      [ "hh::traits::is_included< T1, T2 >", "d0/df0/a00934.html", null ]
    ] ],
    [ "hh::AbstractMemoryManager< ManagedMemory, class >", "dc/d8f/a00718.html", null ],
    [ "hh::AbstractMemoryManager< ManagedMemory >", "dc/d8f/a00718.html", [
      [ "hh::StaticMemoryManager< ManagedMemory, Args >", "d2/db1/a00730.html", null ]
    ] ],
    [ "hh::AbstractMemoryManager< ManagedMemory, typename std::enable_if_t<!traits::is_managed_memory_v< ManagedMemory > > >", "db/d7c/a00722.html", null ],
    [ "hh::AbstractPrinter", "d1/da6/a00738.html", [
      [ "hh::DotPrinter", "d4/d8f/a00742.html", null ]
    ] ],
    [ "hh::AbstractScheduler", "d4/dfa/a00746.html", [
      [ "hh::DefaultScheduler", "d9/db0/a00750.html", null ]
    ] ],
    [ "hh::traits::Contains< T, Ts >", "d0/ddf/a00918.html", null ],
    [ "hh::traits::Contains< T, Ts... >", "d0/ddf/a00918.html", [
      [ "hh::traits::Contains< T, std::tuple< Ts... > >", "dd/dcf/a00922.html", null ]
    ] ],
    [ "hh::core::CoreExecute< NodeInput >", "de/d79/a00798.html", null ],
    [ "hh::core::CoreExecute< TaskInputs >", "de/d79/a00798.html", [
      [ "hh::core::CoreTask< GraphOutput, GraphInputs... >", "d3/d50/a00882.html", [
        [ "hh::core::CoreExecutionPipeline< GraphOutput, GraphInputs... >", "d2/dab/a00886.html", [
          [ "hh::core::CoreDefaultExecutionPipelineExecute< GraphInputs, GraphOutput, GraphInputs... >", "d5/dcd/a00802.html", [
            [ "hh::core::CoreDefaultExecutionPipeline< GraphOutput, GraphInputs >", "d0/dad/a00806.html", null ]
          ] ],
          [ "hh::core::CoreDefaultExecutionPipelineExecute< GraphInput, GraphOutput, GraphInputs >", "d5/dcd/a00802.html", null ]
        ] ],
        [ "hh::core::CoreExecutionPipeline< GraphOutput, GraphInputs >", "d2/dab/a00886.html", null ]
      ] ],
      [ "hh::core::CoreTask< TaskOutput, TaskInputs... >", "d3/d50/a00882.html", [
        [ "hh::core::DefaultCoreTaskExecute< TaskInputs, TaskOutput, TaskInputs... >", "d4/dfd/a00810.html", [
          [ "hh::core::CoreDefaultTask< TaskOutput, TaskInputs >", "da/d71/a00814.html", null ]
        ] ],
        [ "hh::core::DefaultCoreTaskExecute< TaskInput, TaskOutput, TaskInputs >", "d4/dfd/a00810.html", null ]
      ] ],
      [ "hh::core::CoreTask< TaskOutput, TaskInputs >", "d3/d50/a00882.html", null ]
    ] ],
    [ "hh::core::CoreNode", "d5/d69/a00878.html", [
      [ "hh::core::CoreReceiver< GraphInput >", "d6/da7/a00822.html", [
        [ "hh::core::CoreGraphReceiver< GraphInput >", "d0/d11/a00842.html", null ]
      ] ],
      [ "hh::core::CoreReceiver< GraphInputs >", "d6/da7/a00822.html", [
        [ "hh::core::CoreGraphReceiver< GraphInputs >", "d0/d11/a00842.html", [
          [ "hh::core::CoreGraphMultiReceivers< GraphInputs... >", "d2/dc7/a00838.html", [
            [ "hh::core::CoreGraph< GraphOutput, GraphInputs >", "df/d2a/a00874.html", null ]
          ] ],
          [ "hh::core::CoreGraphMultiReceivers< GraphInputs >", "d2/dc7/a00838.html", null ]
        ] ]
      ] ],
      [ "hh::core::CoreReceiver< Inputs >", "d6/da7/a00822.html", [
        [ "hh::core::CoreMultiReceivers< GraphInputs... >", "d3/d60/a00818.html", [
          [ "hh::core::CoreGraphMultiReceivers< GraphInputs... >", "d2/dc7/a00838.html", null ],
          [ "hh::core::CoreGraphMultiReceivers< GraphInputs >", "d2/dc7/a00838.html", null ]
        ] ],
        [ "hh::core::CoreMultiReceivers< NodeInputs... >", "d3/d60/a00818.html", [
          [ "hh::core::CoreQueueMultiReceivers< GraphOutput >", "d6/d2f/a00854.html", [
            [ "hh::core::CoreGraphSink< GraphOutput >", "d9/d87/a00846.html", null ]
          ] ],
          [ "hh::core::CoreQueueMultiReceivers< TaskInputs... >", "d6/d2f/a00854.html", [
            [ "hh::core::CoreTask< GraphOutput, GraphInputs... >", "d3/d50/a00882.html", null ],
            [ "hh::core::CoreTask< TaskOutput, TaskInputs... >", "d3/d50/a00882.html", null ],
            [ "hh::core::CoreTask< TaskOutput, TaskInputs >", "d3/d50/a00882.html", null ]
          ] ],
          [ "hh::core::CoreQueueMultiReceivers< NodeInputs >", "d6/d2f/a00854.html", null ]
        ] ],
        [ "hh::core::CoreMultiReceivers< Inputs >", "d3/d60/a00818.html", null ]
      ] ],
      [ "hh::core::CoreReceiver< NodeInput >", "d6/da7/a00822.html", [
        [ "hh::core::CoreQueueReceiver< NodeInput >", "d0/d66/a00858.html", null ]
      ] ],
      [ "hh::core::CoreReceiver< NodeInputs >", "d6/da7/a00822.html", [
        [ "hh::core::CoreQueueReceiver< NodeInputs >", "d0/d66/a00858.html", [
          [ "hh::core::CoreQueueMultiReceivers< GraphOutput >", "d6/d2f/a00854.html", null ],
          [ "hh::core::CoreQueueMultiReceivers< TaskInputs... >", "d6/d2f/a00854.html", null ],
          [ "hh::core::CoreQueueMultiReceivers< NodeInputs >", "d6/d2f/a00854.html", null ]
        ] ]
      ] ],
      [ "hh::core::CoreNotifier", "d9/daf/a00830.html", [
        [ "hh::core::CoreSender< GraphInputs >", "d9/d02/a00834.html", [
          [ "hh::core::CoreQueueSender< GraphInputs >", "da/d98/a00870.html", [
            [ "hh::core::CoreGraphSource< GraphInputs >", "dc/d0a/a00850.html", null ],
            [ "hh::core::CoreSwitch< GraphInputs >", "d7/d46/a00890.html", null ]
          ] ]
        ] ],
        [ "hh::core::CoreSender< GraphOutput >", "d9/d02/a00834.html", [
          [ "hh::core::CoreQueueSender< GraphOutput >", "da/d98/a00870.html", [
            [ "hh::core::CoreTask< GraphOutput, GraphInputs... >", "d3/d50/a00882.html", null ]
          ] ],
          [ "hh::core::CoreGraph< GraphOutput, GraphInputs >", "df/d2a/a00874.html", null ]
        ] ],
        [ "hh::core::CoreSender< NodeOutput >", "d9/d02/a00834.html", [
          [ "hh::core::CoreQueueSender< NodeOutput >", "da/d98/a00870.html", null ]
        ] ],
        [ "hh::core::CoreSender< TaskOutput >", "d9/d02/a00834.html", [
          [ "hh::core::CoreQueueSender< TaskOutput >", "da/d98/a00870.html", [
            [ "hh::core::CoreTask< TaskOutput, TaskInputs... >", "d3/d50/a00882.html", null ],
            [ "hh::core::CoreTask< TaskOutput, TaskInputs >", "d3/d50/a00882.html", null ]
          ] ]
        ] ],
        [ "hh::core::CoreQueueNotifier", "dd/db2/a00866.html", [
          [ "hh::core::CoreQueueSender< GraphInputs >", "da/d98/a00870.html", null ],
          [ "hh::core::CoreQueueSender< GraphOutput >", "da/d98/a00870.html", null ],
          [ "hh::core::CoreQueueSender< TaskOutput >", "da/d98/a00870.html", null ],
          [ "hh::core::CoreQueueSender< NodeOutput >", "da/d98/a00870.html", null ]
        ] ],
        [ "hh::core::CoreSender< Output >", "d9/d02/a00834.html", null ]
      ] ],
      [ "hh::core::CoreReceiver< Input >", "d6/da7/a00822.html", null ],
      [ "hh::core::CoreSlot", "da/d90/a00826.html", [
        [ "hh::core::CoreMultiReceivers< GraphInputs... >", "d3/d60/a00818.html", null ],
        [ "hh::core::CoreMultiReceivers< NodeInputs... >", "d3/d60/a00818.html", null ],
        [ "hh::core::CoreMultiReceivers< Inputs >", "d3/d60/a00818.html", null ],
        [ "hh::core::CoreQueueSlot", "dc/d41/a00862.html", [
          [ "hh::core::CoreQueueMultiReceivers< GraphOutput >", "d6/d2f/a00854.html", null ],
          [ "hh::core::CoreQueueMultiReceivers< TaskInputs... >", "d6/d2f/a00854.html", null ],
          [ "hh::core::CoreQueueMultiReceivers< NodeInputs >", "d6/d2f/a00854.html", null ]
        ] ]
      ] ]
    ] ],
    [ "enable_shared_from_this", null, [
      [ "hh::MemoryData< ManagedMemory >", "d6/d59/a00726.html", null ]
    ] ],
    [ "hh::behavior::Execute< Input >", "d3/dee/a00774.html", null ],
    [ "hh::behavior::Execute< StateInputs >", "d3/dee/a00774.html", [
      [ "hh::AbstractState< StateOutput, StateInputs >", "d8/d4e/a00754.html", null ]
    ] ],
    [ "hh::behavior::Execute< TaskInputs >", "d3/dee/a00774.html", [
      [ "hh::AbstractTask< GraphOutput, TaskInputs... >", "d4/deb/a00710.html", null ],
      [ "hh::AbstractTask< StateOutput, StateInputs... >", "d4/deb/a00710.html", [
        [ "hh::behavior::BaseStateManager< StateOutput, StateInputs... >", "d2/d4a/a00758.html", [
          [ "hh::behavior::StateManagerExecuteDefinition< StateInput, StateOutput, StateInputs >", "df/d5f/a00762.html", null ],
          [ "hh::behavior::StateManagerExecuteDefinition< StateInputs, StateOutput, StateInputs... >", "df/d5f/a00762.html", [
            [ "hh::StateManager< StateOutput, StateInputs >", "d4/d13/a00766.html", null ]
          ] ]
        ] ],
        [ "hh::behavior::BaseStateManager< StateOutput, StateInputs >", "d2/d4a/a00758.html", null ]
      ] ],
      [ "hh::AbstractTask< TaskOutput, TaskInputs... >", "d4/deb/a00710.html", [
        [ "hh::AbstractCUDATask< TaskOutput, TaskInputs >", "dd/d85/a00702.html", null ]
      ] ],
      [ "hh::AbstractTask< TaskOutput, TaskInputs >", "d4/deb/a00710.html", null ]
    ] ],
    [ "hh::GraphSignalHandler< GraphOutput, GraphInputs >", "d0/d7e/a00770.html", null ],
    [ "hh::StaticMemoryManager< ManagedMemory, Args >::HasConstructor", "d6/dc0/a00734.html", null ],
    [ "hh::helper::HelperCoreMultiReceiversType< Inputs >", "d5/dd3/a00902.html", null ],
    [ "hh::helper::HelperCoreMultiReceiversType< std::tuple< Inputs... > >", "da/dd8/a00906.html", null ],
    [ "hh::helper::HelperMultiReceiversType< Inputs >", "dc/dfa/a00894.html", null ],
    [ "hh::helper::HelperMultiReceiversType< std::tuple< Inputs... > >", "d2/d79/a00898.html", null ],
    [ "hh::traits::IsManagedMemory< PossibleManagedMemory >", "d3/dd1/a00914.html", null ],
    [ "hh::behavior::Node", "d3/daf/a00790.html", [
      [ "hh::AbstractTask< GraphOutput, TaskInputs... >", "d4/deb/a00710.html", null ],
      [ "hh::AbstractTask< StateOutput, StateInputs... >", "d4/deb/a00710.html", null ],
      [ "hh::AbstractTask< TaskOutput, TaskInputs... >", "d4/deb/a00710.html", null ],
      [ "hh::Graph< GraphOutput, GraphInputs... >", "df/d96/a00714.html", null ],
      [ "hh::AbstractTask< TaskOutput, TaskInputs >", "d4/deb/a00710.html", null ],
      [ "hh::behavior::MultiReceivers< Inputs >", "d6/d17/a00778.html", null ],
      [ "hh::behavior::Sender< Output >", "da/d96/a00782.html", null ],
      [ "hh::Graph< GraphOutput, GraphInputs >", "df/d96/a00714.html", null ],
      [ "hh::behavior::MultiReceivers< GraphInputs... >", "d6/d17/a00778.html", [
        [ "hh::AbstractExecutionPipeline< GraphOutput, GraphInputs... >", "d8/dcb/a00706.html", null ],
        [ "hh::Graph< GraphOutput, GraphInputs... >", "df/d96/a00714.html", null ],
        [ "hh::AbstractExecutionPipeline< GraphOutput, GraphInputs >", "d8/dcb/a00706.html", null ],
        [ "hh::Graph< GraphOutput, GraphInputs >", "df/d96/a00714.html", null ]
      ] ],
      [ "hh::behavior::MultiReceivers< TaskInputs... >", "d6/d17/a00778.html", [
        [ "hh::AbstractTask< GraphOutput, TaskInputs... >", "d4/deb/a00710.html", null ],
        [ "hh::AbstractTask< StateOutput, StateInputs... >", "d4/deb/a00710.html", null ],
        [ "hh::AbstractTask< TaskOutput, TaskInputs... >", "d4/deb/a00710.html", null ],
        [ "hh::AbstractTask< TaskOutput, TaskInputs >", "d4/deb/a00710.html", null ]
      ] ],
      [ "hh::behavior::Sender< GraphOutput >", "da/d96/a00782.html", [
        [ "hh::AbstractExecutionPipeline< GraphOutput, GraphInputs... >", "d8/dcb/a00706.html", null ],
        [ "hh::AbstractTask< GraphOutput, TaskInputs... >", "d4/deb/a00710.html", null ],
        [ "hh::Graph< GraphOutput, GraphInputs... >", "df/d96/a00714.html", null ],
        [ "hh::AbstractExecutionPipeline< GraphOutput, GraphInputs >", "d8/dcb/a00706.html", null ],
        [ "hh::Graph< GraphOutput, GraphInputs >", "df/d96/a00714.html", null ]
      ] ],
      [ "hh::behavior::Sender< StateOutput >", "da/d96/a00782.html", [
        [ "hh::AbstractTask< StateOutput, StateInputs... >", "d4/deb/a00710.html", null ]
      ] ],
      [ "hh::behavior::Sender< TaskOutput >", "da/d96/a00782.html", [
        [ "hh::AbstractTask< TaskOutput, TaskInputs... >", "d4/deb/a00710.html", null ],
        [ "hh::AbstractTask< TaskOutput, TaskInputs >", "d4/deb/a00710.html", null ]
      ] ]
    ] ],
    [ "hh::NvtxProfiler", "d4/d12/a00910.html", null ],
    [ "hh::behavior::Pool< ManagedData >", "d0/d7d/a00786.html", null ],
    [ "hh::behavior::Pool< ManagedMemory >", "d0/d7d/a00786.html", null ],
    [ "hh::behavior::SwitchRule< GraphInput >", "d5/d12/a00794.html", null ],
    [ "hh::behavior::SwitchRule< GraphInputs >", "d5/d12/a00794.html", [
      [ "hh::AbstractExecutionPipeline< GraphOutput, GraphInputs... >", "d8/dcb/a00706.html", null ],
      [ "hh::AbstractExecutionPipeline< GraphOutput, GraphInputs >", "d8/dcb/a00706.html", null ]
    ] ],
    [ "bool", "d5/d4c/a01058.html", null ],
    [ "condition_variable", "d7/d8c/a01306.html", null ],
    [ "int", "df/db5/a01158.html", null ],
    [ "shared_ptr< hh::AbstractMemoryManager< GraphOutput > >", "db/d32/a01122.html", null ],
    [ "shared_ptr< hh::AbstractMemoryManager< StateOutput > >", "d4/d22/a01250.html", null ],
    [ "shared_ptr< hh::AbstractTask< GraphOutput, TaskInputs... > >", "d0/d61/a01130.html", null ],
    [ "shared_ptr< hh::AbstractTask< TaskOutput, TaskInputs... > >", "d0/dfb/a01066.html", null ],
    [ "shared_ptr< hh::core::CoreTask< GraphOutput, TaskInputs... > >", "dd/d09/a01118.html", null ],
    [ "shared_ptr< hh::core::CoreTask< StateOutput, TaskInputs... > >", "d7/d04/a01246.html", null ],
    [ "shared_ptr< ManagedMemory >", "d7/d88/a01294.html", null ],
    [ "shared_ptr< std::queue< std::shared_ptr< NodeInputs > > >", "d3/d08/a01018.html", null ],
    [ "shared_ptr< std::set< hh::core::CoreQueueReceiver< GraphInputs > * > >", "dc/d19/a01190.html", null ],
    [ "shared_ptr< std::set< hh::core::CoreQueueReceiver< GraphOutput > * > >", "da/d4e/a01106.html", null ],
    [ "shared_ptr< std::set< hh::core::CoreQueueReceiver< TaskOutput > * > >", "d9/d9a/a00998.html", null ],
    [ "shared_ptr< std::set< hh::core::CoreSender< NodeInputs > * > >", "d1/dd8/a01022.html", null ],
    [ "size_t", "de/d02/a01026.html", null ]
];