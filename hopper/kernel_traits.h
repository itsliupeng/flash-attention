/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVO {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    union {
        cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
    struct {
        cutlass::arch::ClusterTransactionBarrier barrier_Q;
        cutlass::arch::ClusterBarrier barrier_O;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        int tile_count_semaphore;
    };
};

template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVOVt {
  struct {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    // cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;  
    union {
        cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v_out;
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
  };
  struct {    
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    // typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    typename cutlass::PipelineAsync<kStages>::SharedStorage pipeline_vt;
    int tile_count_semaphore;
  };
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int kStages_, bool Is_Q_in_regs_=false,
         int kClusterM_ = 1, typename elem_type=cutlass::half_t>
struct Flash_fwd_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;
    using OutputType = elem_type;
    using index_t = int64_t;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_; // 12
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp; // 12 * 32 = 384
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp; // 32

    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_;
    static_assert(kNWarps_ == 4 || kNWarps_ == 8 || kNWarps_ == 12 || kNWarps_ == 16);
    static constexpr bool Is_WS = kNWarps_ >= 12;
    static_assert(!(Is_WS && Is_Q_in_regs), "Warp-specialization does not support Q in registers");

    static constexpr int kBlockM = kBlockM_; // 128
    static constexpr int kBlockN = kBlockN_; // 80
    static constexpr int kHeadDim = kHeadDim_; // 256
    static_assert(kHeadDim % 32 == 0);
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    static constexpr int kClusterM = kClusterM_; // 1
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    static constexpr int kStages = kStages_; // 2

    using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>; // (2, 1, 1)
    using TiledMma0 = decltype(cute::make_tiled_mma(
        std::conditional_t<
            Is_Q_in_regs,
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK>()),
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>())
        >{},
        AtomLayoutMNK{}));
    using TiledMma1 = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, decltype(select<0, 2, 1>(TileShape_MNK{})),
                                   GMMA::Major::K, GMMA::Major::MN>(),
        AtomLayoutMNK{}));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    // SmemLayoutQ: Sw<3,4,3> o smem_ptr[16b](unset) o (_128,(_64,_4)):(_64,(_1,_8192))
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    // SmemLayoutK: Sw<3,4,3> o smem_ptr[16b](unset) o (_80,(_64,_4),_2):(_64,(_1,_5120),_20480)
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    // SmemLayoutV: Sw<3,4,3> o smem_ptr[16b](unset) o (_80,(_64,_4),_2):(_64,(_1,_5120),_20480)
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), Int<kStages>{})));

    // Note this is the transpose in terms of the view, not in terms of memory.
    // SmemLayoutVt: Sw<3,4,3> o smem_ptr[16b](unset) o ((_64,_4),_80,_2):((_1,_5120),_64,_20480)
    using SmemLayoutVt =
        decltype(composition(SmemLayoutV{},
                    make_ordered_layout(
                        make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{}), Int<kStages>{}),
                        Step<_2, _1, _3>{})));

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, OutputType,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    // SmemLayoutO: Sw<3,4,3> o smem_ptr[16b](unset) o (_128,(_64,_4)):(_64,(_1,_8192))
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

    using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

    using SharedStorage = SharedStorageQKVO<kStages, Element, Element, Element, SmemLayoutQ,
                                            SmemLayoutK, SmemLayoutV, SmemLayoutO>;

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    // using BarrierType = typename MainloopPipeline::ProducerBarrierType;

    void print() const {
        using namespace cute;
        std::cout << "in Flash_fwd_kernel_traits >>>>>" << std::endl;
        cute::print("\t kNWarps: "); cute::print(kNWarps); cute::print("\n");
        cute::print("\t kNThreads: "); cute::print(kNThreads); cute::print("\n");
        cute::print("\t NumProducerThreads: "); cute::print(NumProducerThreads); cute::print("\n");
        cute::print("\t cutlass::NumThreadsPerWarpGroup: "); cute::print(cutlass::NumThreadsPerWarpGroup); cute::print("\n");
        cute::print("\t Is_Q_in_regs: "); cute::print(Is_Q_in_regs); cute::print("\n");
        cute::print("\t kBlockM: "); cute::print(kBlockM); cute::print("\n");
        cute::print("\t kBlockN: "); cute::print(kBlockN); cute::print("\n");
        cute::print("\t kHeadDim: "); cute::print(kHeadDim); cute::print("\n");
        cute::print("\t kClusterM: "); cute::print(kClusterM); cute::print("\n");
        cute::print("\t kStages: "); cute::print(kStages); cute::print("\n");
        cute::print("\t SmemLayoutQ: "); cute::print(SmemLayoutQ{}); cute::print("\n");
        cute::print("\t SmemLayoutK: "); cute::print(SmemLayoutK{}); cute::print("\n");
        cute::print("\t SmemLayoutV: "); cute::print(SmemLayoutV{}); cute::print("\n");
        cute::print("\t SmemLayoutVt: "); cute::print(SmemLayoutVt{}); cute::print("\n");
        cute::print("\t SmemLayoutO: "); cute::print(SmemLayoutO{}); cute::print("\n");
        cute::print("\t TiledMma0: "); cute::print(TiledMma0{}); cute::print("\n");
        cute::print("\t TiledMma1: "); cute::print(TiledMma1{}); cute::print("\n");
        // Continue for other fields as necessary...
        std::cout << "<<<<< in Flash_fwd_kernel_traits" << std::endl;
    }

};

// Traits struct for fp8 kernel with in-kernel transpose
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int kStages_, bool Is_Q_in_regs_=false,
         int kClusterM_ = 1, typename elem_type=cutlass::float_e4m3_t>
struct Flash_fwd_kernel_traits_fp8 {
    using Element = elem_type;
    static_assert(cutlass::sizeof_bits_v<Element> == 8);
    using ElementAccum = float;
    using OutputType = cutlass::half_t;
    using index_t = int64_t;      

    // The number of threads.
    static constexpr int kNWarps = kNWarps_; // 12
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp; 
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarpGroup; // 128

    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_;
    static_assert(kNWarps_ == 12 || kNWarps_ == 16);
    static constexpr bool Is_WS = true;    
    static_assert(!Is_Q_in_regs, "Warp-specialization does not support Q in registers");    

    static constexpr int kBlockM = kBlockM_; // 128
    static constexpr int kBlockN = kBlockN_; // 128
    static constexpr int kHeadDim = kHeadDim_; // 256
    static_assert(kHeadDim % 32 == 0);
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

    static constexpr int kClusterM = kClusterM_; // 1
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    static constexpr int kStages = kStages_; // 2
    static_assert(kStages > 1);

    using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;  // (2, 1, 1)
    // TiledMma0: TiledMMA
    //     ThrLayoutVMNK:  (_128,_2,_1,_1):(_1,_128,_0,_0)
    //     PermutationMNK: (_,_,_)
    //     MMA_Atom
    //     ThrID:      _128:_1
    //     LayoutA_TV: (_128,(_64,_32)):(_0,(_1,_64))
    //     LayoutB_TV: (_128,(_128,_32)):(_0,(_1,_128))
    //     LayoutC_TV: ((_4,_8,_4),(_2,_2,_16)):((_128,_1,_16),(_64,_8,_512))
    using TiledMma0 = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutMNK{}));
    
    // TiledMma1: TiledMMA
    //     ThrLayoutVMNK:  (_128,_2,_1,_1):(_1,_128,_0,_0)
    //     PermutationMNK: (_,_,_)
    //     MMA_Atom
    //     ThrID:      _128:_1
    //     LayoutA_TV: ((_4,_8,_4),(_4,_2,_2)):((_256,_1,_16),(_64,_8,_1024))
    //     LayoutB_TV: (_128,(_256,_32)):(_0,(_1,_256))
    //     LayoutC_TV: ((_4,_8,_4),(_2,_2,_32)):((_128,_1,_16),(_64,_8,_512))
    using TiledMma1 = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, decltype(select<0, 2, 1>(TileShape_MNK{}))>(),
        AtomLayoutMNK{}));

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    // Sw<3,4,3> o smem_ptr[8b](unset) o (_128,(_128,_2)):(_128,(_1,_16384))
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    using TransposeShapeAtomV = Shape<_64, _64>;    
    using SmemLayoutAtomV = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}));
    // Sw<2,4,3> o smem_ptr[8b](unset) o (_128,(_64,_4),_2):(_64,(_1,_8192),_32768)
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{}))); // (N, K, 2)

    // using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
    //     decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>()); // (N, K)
    // // Sw<3,4,3> o smem_ptr[8b](unset) o (_128,(_128,_2),_2):(_128,(_1,_16384),_32768)
    // using SmemLayoutK =
    //     decltype(tile_to_shape(SmemLayoutAtomK{},
    //              make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutK = SmemLayoutV;
    // for fp8 in-kernel transpose -- src layout
    // Sw<2,4,3> o smem_ptr[8b](unset) o ((_64,_64),_2,_4,_2):((_64,_1),_4096,_8192,_32768)
    using SmemLayoutDivideV = decltype(tiled_divide(SmemLayoutV{}, TransposeShapeAtomV{}));
    using SmemShapeLDSM = Shape<Shape<_8, _8>, Shape<_16, _4>>; // 128 * 2 / 8 = 32 (8x8 fp16) = 64 (8x8 fp8)
    // (((_8,_8),(_16,_4)),_2,_4,_2)
    using FactoringShapeV = decltype(make_shape(SmemShapeLDSM{},
        shape<1>(SmemLayoutDivideV{}), shape<2>(SmemLayoutDivideV{}), shape<3>(SmemLayoutDivideV{})));
    // Sw<2,4,3> o smem_ptr[8b](unset) o (((_8,_8),(_16,_4)),_2,_4,_2):(((_64,_512),(_1,_16)),_4096,_8192,_32768)
    using SmemLayoutTransposeV = decltype(composition(SmemLayoutDivideV{}, make_layout(FactoringShapeV{})));

    // For fp8, this is the memory transpose.
    // Sw<2,4,3> o smem_ptr[8b](unset) o (_64,_64):(_64,_1)
    using SmemLayoutAtomVt = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}));
    // Sw<2,4,3> o smem_ptr[8b](unset) o (_256,(_64,_2),_2):(_64,(_1,_16384),_32768)
    using SmemLayoutVt =
        decltype(tile_to_shape(SmemLayoutAtomVt{},
                 make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{}))); // (K, N, 2)

    // for fp8 in-kernel transpose -- dst layout
    // Sw<2,4,3> o smem_ptr[8b](unset) o ((_64,_2),_256,_2):((_1,_16384),_64,_32768)
    using SmemLayoutVtTrans =
        decltype(composition(SmemLayoutVt{},
                             make_ordered_layout(product_each(shape(SmemLayoutV{})), Step<_2, _1, _3>{}))); // (K, N, 2)
    using SmemLayoutDivideVt = decltype(tiled_divide(SmemLayoutVtTrans{}, TransposeShapeAtomV{}));
#ifndef NO_FP8_COLUMN_PERMUTE
    using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_8, _8>>; // hit
#else
    using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_16, _4>>;
#endif
    using FactoringShapeVt = decltype(make_shape(SmemShapeSTSM{},
        shape<1>(SmemLayoutDivideVt{}), shape<2>(SmemLayoutDivideVt{}), shape<3>(SmemLayoutDivideVt{})));
    using SmemLayoutTransposeVt = decltype(composition(SmemLayoutDivideVt{}, make_layout(FactoringShapeVt{})));

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, OutputType,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

    // used for rmem -> smem O copy in fp8 kernel to undo column permutation
    using ThreadLayoutrO = Layout<Shape<_8, Int<kBlockM/16>, _4, _1>,
                                 Stride<_4, _32, _1, _0>>;
    using ValueLayoutrO = Layout<Shape<_1, _2, Shape<_2, _2>, Int<kHeadDim/16>>,
                                Stride<_0, _2, Stride<_4, _1>, _8>>;
    using TiledCopyrO = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint16_t>, OutputType>{},
                      ThreadLayoutrO{}, ValueLayoutrO{}));

    using TiledCopyShaperO = Shape<_8, Int<kBlockM/8>, _16, Int<kHeadDim/16>>;
    using SmemLayoutrO = decltype(composition(SmemLayoutO{}, Layout<TiledCopyShaperO>{}));

    using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

    using SharedStorage = SharedStorageQKVOVt<kStages, Element, Element, OutputType, SmemLayoutQ,
                          SmemLayoutK, SmemLayoutV, SmemLayoutO>;

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    // using BarrierType = typename MainloopPipeline::ProducerBarrierType;

    void print() const {
        std::cout << "in Flash_fwd_kernel_traits_fp8 >>>>>" << std::endl;
        cute::print("\t kNWarps: "); cute::print(kNWarps); cute::print("\n");
        cute::print("\t kNThreads: "); cute::print(kNThreads); cute::print("\n");
        cute::print("\t NumProducerThreads: "); cute::print(NumProducerThreads); cute::print("\n");
        cute::print("\t Is_Q_in_regs: "); cute::print(Is_Q_in_regs); cute::print("\n");
        cute::print("\t kBlockM: "); cute::print(kBlockM); cute::print("\n");
        cute::print("\t kBlockN: "); cute::print(kBlockN); cute::print("\n");
        cute::print("\t kHeadDim: "); cute::print(kHeadDim); cute::print("\n");
        cute::print("\t kClusterM: "); cute::print(kClusterM); cute::print("\n");
        cute::print("\t kStages: "); cute::print(kStages); cute::print("\n");
        cute::print("\t SmemLayoutQ: "); cute::print(SmemLayoutQ{}); cute::print("\n");
        cute::print("\t SmemLayoutK: "); cute::print(SmemLayoutK{}); cute::print("\n");
        cute::print("\t SmemLayoutV: "); cute::print(SmemLayoutV{}); cute::print("\n");
        cute::print("\t SmemLayoutDivideV: "); cute::print(SmemLayoutDivideV{}); cute::print("\n");
        cute::print("\t FactoringShapeV: "); cute::print(FactoringShapeV{}); cute::print("\n");
        cute::print("\t SmemLayoutTransposeV: "); cute::print(SmemLayoutTransposeV{}); cute::print("\n");
        cute::print("\t SmemLayoutAtomVt: "); cute::print(SmemLayoutAtomVt{}); cute::print("\n");
        cute::print("\t SmemLayoutVt: "); cute::print(SmemLayoutVt{}); cute::print("\n");
        cute::print("\t SmemLayoutVtTrans: "); cute::print(SmemLayoutVtTrans{}); cute::print("\n");
        cute::print("\t SmemLayoutDivideVt: "); cute::print(SmemLayoutDivideVt{}); cute::print("\n");
        cute::print("\t FactoringShapeVt: "); cute::print(FactoringShapeVt{}); cute::print("\n");
        cute::print("\t SmemLayoutTransposeVt: "); cute::print(SmemLayoutTransposeVt{}); cute::print("\n");
        cute::print("\t SmemLayoutO: "); cute::print(SmemLayoutO{}); cute::print("\n");
        cute::print("\t TiledMma0: "); cute::print(TiledMma0{}); cute::print("\n");
        cute::print("\t TiledMma1: "); cute::print(TiledMma1{}); cute::print("\n");
        // Continue for other fields as necessary...
        std::cout << "<<<<< in Flash_fwd_kernel_traits_fp8" << std::endl;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
