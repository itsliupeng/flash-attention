/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "utils.h"


#define PRINT_DEBUG_SITE() \
    printf("current PC at File: %s, Line: %d, Function: %s\n", __FILE__, __LINE__, __func__)

namespace flash {

using namespace cute;

// 4 warps
struct SmemTransposeFp8_64x64 {

  using Element = cutlass::float_e4m3_t;
  
  using ldsm_thread_shape = Shape<_4, _1, _8, _4>; // 32 * 4
  using ldsm_value_shape = Shape<_2, _8, _2, _1>;   // 16 * 2
  using ldsm_value_stride = Stride<_2, _4, _1, _0>;
  using TiledCopyLDSM = decltype(make_tiled_copy(
      Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, Layout<ldsm_thread_shape>{},
      Layout<ldsm_value_shape, ldsm_value_stride>{}));
  TiledCopyLDSM tiled_copy_ldsm;  

  using stsm_thread_shape = Shape<_4, _1, _8, _4>; // 32 * 4
  // using stsm_thread_stride = Stride<_1, _0, _4, _32>;
#ifndef NO_FP8_COLUMN_PERMUTE
  using stsm_value_shape = Shape<_4, _4, _1, _2>; // 16 * 2
  using stsm_value_stride = Stride<_1, _8, _0, _4>;
#else
  using stsm_value_shape = Shape<_4, _4, _2, _1>;
  using stsm_value_stride = Stride<_1, _8, _4, _0>;
#endif

  using TiledCopySTSM =
      decltype(make_tiled_copy(Copy_Atom<SM90_U32x4_STSM_N, Element>{},
                               Layout<stsm_thread_shape>{},
                               Layout<stsm_value_shape, stsm_value_stride>{}));
  TiledCopySTSM tiled_copy_stsm;

  template <class SmemTensor, class SmemTensorOut>
  CUTLASS_DEVICE void operator()(SmemTensor &&s_in, SmemTensorOut &&s_out) {
    using namespace cute;

    auto tid = threadIdx.x;
    auto thr_copy_ldsm = tiled_copy_ldsm.get_thread_slice(tid);
    auto thr_copy_stsm = tiled_copy_stsm.get_thread_slice(tid);

    auto tXsX = thr_copy_ldsm.partition_S(s_in);
    auto tXrX = make_tensor<Element>(shape(tXsX));    
    auto tXsX_out = thr_copy_stsm.partition_D(s_out);

    cute::copy(tiled_copy_ldsm, tXsX, tXrX);

    auto data = tXrX.data();
    // size(tXrX) == 32
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size(tXrX); n += 8) {
      uint32_t *data_32bit = reinterpret_cast<uint32_t *>(&data[n]);
      auto upper = data_32bit[0];
      auto lower = data_32bit[1];
      data_32bit[0] = __byte_perm(upper, lower, 0x6420);
      data_32bit[1] = __byte_perm(upper, lower, 0x7531);
    }

    cute::copy(tiled_copy_stsm, tXrX, tXsX_out);
  }
};

template <typename Ktraits, bool Is_causal, typename Seqlen_traits>
struct CollectiveMainloopFwd {

    using Element = typename Ktraits::Element;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;

    static constexpr int kStages = Ktraits::kStages;
    static constexpr int kHeadDim = Ktraits::kHeadDim;    

    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));
    
    using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
    using SmemLayoutK = typename Ktraits::SmemLayoutK;
    using SmemLayoutV = typename Ktraits::SmemLayoutV;
    using SmemLayoutVt = typename Ktraits::SmemLayoutVt;

    using TMA_Q = decltype(make_tma_copy(
        GmemTiledCopyQ{},
        make_tensor(
            make_gmem_ptr(static_cast<Element const*>(nullptr)), 
            repeat_like(typename Seqlen_traits::StrideT{}, int32_t(0)), 
            typename Seqlen_traits::StrideT{}
        ),
        SmemLayoutQ{},
        select<0, 2>(TileShape_MNK{}),
        _1{}));  // no mcast for Q

    using TMA_K = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(
            make_gmem_ptr(static_cast<Element const*>(nullptr)), 
            repeat_like(typename Seqlen_traits::StrideT{}, int32_t(0)), 
            typename Seqlen_traits::StrideT{}
        ),
        take<0, 2>(SmemLayoutK{}),
        select<1, 2>(TileShape_MNK{}),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

    // TMA_V may differ from TMA_K for fp8 kernel (e.g. swizzling mode)
    using TMA_V = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(
            make_gmem_ptr(static_cast<Element const*>(nullptr)),
            repeat_like(typename Seqlen_traits::StrideT{}, int32_t(0)),
            typename Seqlen_traits::StrideT{}
        ),
        take<0, 2>(SmemLayoutK{}),
        select<1, 2>(TileShape_MNK{}),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

    static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using MainloopPipelineNoTMA = typename Ktraits::MainloopPipelineNoTMA;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);

    // static constexpr bool UseSchedulerBarrier = kHeadDim <= 128;
    static constexpr bool UseSchedulerBarrier =
        cutlass::sizeof_bits_v<Element> == 8 ? kHeadDim >= 128
                                             : kHeadDim <= 128;    

    // Host side kernel arguments
    struct Arguments {
        Element const* ptr_Q;
        typename Seqlen_traits::LayoutT layout_Q;
        Element const* ptr_K;
        typename Seqlen_traits::LayoutT layout_K;
        Element const* ptr_V;
        typename Seqlen_traits::LayoutT layout_V;
        cute::TmaDescriptor* tensormaps;
        float const softmax_scale_log2;
        int* block_table;
        int64_t block_table_batch_stride;
        int page_size;
        int64_t page_stride;
        int64_t row_stride;
    };

    // Device side kernel params
    struct Params {
        typename Seqlen_traits::LayoutT layout_Q;
        typename Seqlen_traits::LayoutT layout_K;
        typename Seqlen_traits::LayoutT layout_V;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_Q tma_load_Q;        
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        cute::TmaDescriptor* tensormaps;
        float const softmax_scale_log2;
        Element const* ptr_K;
        int* block_table;
        int64_t block_table_batch_stride;
        int page_size;
        int64_t page_stride;
        int64_t row_stride;

        void print() const {
            cute::print(">>>>> in CollectiveMainloopFwd#Params\n");
            cute::print("\t layout_Q: "); cute::print(layout_Q); cute::print("\n"); // (128,256,16,8):(4096,_1,256,524288)
            cute::print("\t layout_K: "); cute::print(layout_K); cute::print("\n"); // (128,256,16,8):(4096,_1,256,524288)
            cute::print("\t layout_V: "); cute::print(layout_V); cute::print("\n"); // (128,256,16,8):(4096,_1,256,524288)
            cute::print("\t tma_load_Q: "); cute::print(tma_load_Q); cute::print("\n");
            cute::print("\t tma_load_K: "); cute::print(tma_load_K); cute::print("\n");
            cute::print("\t tma_load_V: "); cute::print(tma_load_V); cute::print("\n");
            cute::print("\t block_table: "); cute::print(block_table); cute::print("\n");
            cute::print("\t block_table_batch_stride: "); cute::print(block_table_batch_stride); cute::print("\n");
            cute::print("\t page_size: "); cute::print(page_size); cute::print("\n");
            cute::print("\t page_stride: "); cute::print(page_stride); cute::print("\n");
            cute::print("\t row_stride: "); cute::print(row_stride); cute::print("\n");
            cute::print("<<<<< in CollectiveMainloopFwd#Params\n");  
        };
    };

    CUTLASS_DEVICE cute::TmaDescriptor*
    load_init(Params const& params, int32_t const sm_idx) const {
        cute::TmaDescriptor* gmem_tensormaps = reinterpret_cast<cute::TmaDescriptor*>(params.tensormaps);
        cute::TmaDescriptor* tma_desc = &gmem_tensormaps[sm_idx];

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // Initialize tma for loading
        if ((warp_idx_in_warpgroup == 0) && cute::elect_one_sync()) {
            // Bringing tensormaps from params to gmem for modification later
            Tensor pC_tensormap = make_tensor(params.tma_load_K.get_tma_descriptor(), Int<1>{}, Int<1>{});
            Tensor gC_tensormap = make_tensor(tma_desc, Int<1>{}, Int<1>{});
            copy(recast<uint128_t>(pC_tensormap), recast<uint128_t>(gC_tensormap));
            cp_async_fence();
            cp_async_wait<0>();
        }
        __syncwarp();
        return tma_desc;
    }

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.layout_Q);
        TMA_Q tma_load_Q = make_tma_copy(
            GmemTiledCopyQ{},
            mQ,
            SmemLayoutQ{},
            select<0, 2>(TileShape_MNK{}),
            _1{}); // no mcast for Q
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.layout_K); // [b, n, s, d]
        TMA_K tma_load_K = make_tma_copy(
            GmemTiledCopyKV{},
            mK,
            SmemLayoutK{}(_, _, _0{}),
            select<1, 2>(TileShape_MNK{}),
            size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.layout_V);
        TMA_V tma_load_V = make_tma_copy(
            GmemTiledCopyKV{},
            mV,
            SmemLayoutK{}(_, _, _0{}),
            select<1, 2>(TileShape_MNK{}),
            size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
        return {args.layout_Q, args.layout_K, args.layout_V,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.layout_Q.shape()), get<2>(args.layout_K.shape()))),
                tma_load_Q, tma_load_K, tma_load_V,
                args.tensormaps,
                args.softmax_scale_log2,
                args.ptr_K,
                args.block_table,
                args.block_table_batch_stride,
                args.page_size,
                args.page_stride,
                args.row_stride};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params) {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_V.get_tma_descriptor());
    }

    CUTLASS_DEVICE
    int get_n_block_max(
          Params const& mainloop_params, int m_block, 
          const Seqlen_traits& seqlen_traits_q,
          const Seqlen_traits& seqlen_traits_k,
          bool is_page_cache=false
        ) {
        static constexpr int kBlockM = get<0>(TileShape_MNK{}); // 128
        static constexpr int kBlockN = get<1>(TileShape_MNK{}); // 64 
        int const seqlen_q = Seqlen_traits::kUseVarSeqLen ? seqlen_traits_q.actual_seq_len : shape<0>(mainloop_params.layout_Q);
        int seqlen_k = (Seqlen_traits::kUseVarSeqLen || is_page_cache) ? seqlen_traits_k.actual_seq_len : shape<0>(mainloop_params.layout_K);  
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal) {
            n_block_max = std::min(n_block_max,
                                   cute::ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q, kBlockN));
        }
        return n_block_max;
    }

    template <typename Scheduler, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& mainloop_params,
         MainloopPipeline pipeline_k,
         MainloopPipeline pipeline_v,
         PipelineState& smem_pipe_write_k,
         PipelineState& smem_pipe_write_v,
         SharedStorage &shared_storage,
         Scheduler& scheduler,
         typename Scheduler::Params const& scheduler_params,
         typename Scheduler::WorkTileInfo& work_tile_info,
         cute::tuple<int32_t, int32_t, int32_t> block_coord,
         int work_idx,
         const Seqlen_traits& seqlen_traits_q,
         const Seqlen_traits& seqlen_traits_k
         ) {

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
        // Tensor sK = sV;

        Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());
        // Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.layout_K.shape());
        Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.layout_V.shape()); // (s, h, n, b)
        Tensor mK = mV;

        // print("mK: "); print(mK); print("\n");
        // print("sK: "); print(sK); print("\n");
        // print(cute::thread0());

        auto [m_block, bidh, bidb] = block_coord;
        int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        Tensor gQ = seqlen_traits_q.get_local_tile_tensor(
            mQ, select<0, 2>(TileShape_MNK{}), bidh, bidb)(_, _, m_block);  // (M, K)
        Tensor gK = seqlen_traits_k.get_local_tile_tensor(
            mK, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb);  // (N, K, _)
        Tensor gV = seqlen_traits_k.get_local_tile_tensor(
            mV, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb);  // (N, K, _)

        Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
        Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
        auto [tQgQ, tQsQ] = tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sQ_x), group_modes<0, 2>(gQ_x));  // (TMA), (TMA)
        auto [tKgK, tKsK] = tma_partition(mainloop_params.tma_load_K, block_rank_in_cluster, Layout<ClusterShape>{},
                                          group_modes<0, 2>(sK), group_modes<0, 2>(gK));  // (TMA, k), (TMA, PIPE)
        auto [tVgV, tVsV] = tma_partition(mainloop_params.tma_load_V, block_rank_in_cluster, Layout<ClusterShape>{},
                                          group_modes<0, 2>(sV), group_modes<0, 2>(gV));  // (TMA, k), (TMA, PIPE)

        uint16_t mcast_mask_kv = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int m = 0; m < size<0>(block_layout); ++m) {
                mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
            }
        }

        int n_block_max = get_n_block_max(mainloop_params, m_block, seqlen_traits_q, seqlen_traits_k);
        int n_block = n_block_max - 1;

        int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
            pipeline_k.producer_acquire(smem_pipe_write_k);
            copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
            ++smem_pipe_write_k;
        }

        // Wait for the MMA warpgroups to say that smem_q is ready
        cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);

        if (lane_predicate) {
            shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
            copy(mainloop_params.tma_load_Q.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_Q), 0 /*mcast_mask*/), tQgQ, tQsQ);
        }

        // Wait for warp 1 to signal that smem_v are ready and V can be copied from gmem
        // Need ClusterBarrier, not just NamedBarrier. Otherwise we might have CTA 0 finishing the
        // TMA store on O first, call TMA multicast load on V, before CTA 1 can finishing TMA store on O.
        shared_storage.barrier_O.wait((work_idx + 1) % 2);

        if (lane_predicate) {
            // CUTLASS_PRAGMA_NO_UNROLL
            #pragma unroll 2
            for (; n_block > 0; --n_block) {
                pipeline_k.producer_acquire(smem_pipe_write_k);
                copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                    tKgK(_, n_block - 1), tKsK(_, smem_pipe_write_k.index()));
                ++smem_pipe_write_k;
                pipeline_v.producer_acquire(smem_pipe_write_v);
                copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                    tVgV(_, n_block), tVsV(_, smem_pipe_write_v.index()));
                ++smem_pipe_write_v;
            }
        }
        scheduler.prefetch_next_work(scheduler_params, work_tile_info);
        if (lane_predicate) {
            pipeline_v.producer_acquire(smem_pipe_write_v);
            copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                tVgV(_, n_block), tVsV(_, smem_pipe_write_v.index()));
            ++smem_pipe_write_v;
        }
        scheduler.broadcast_next_work(work_tile_info);
    }

    template <typename Scheduler, typename SharedStorage>
    CUTLASS_DEVICE void
    load_fp8(Params const& mainloop_params,
         cute::TmaDescriptor* tma_load_K_page_ptr,
         const int *block_table,
         MainloopPipeline pipeline_k,
         MainloopPipeline pipeline_v,
         MainloopPipelineNoTMA pipeline_vt,         
         PipelineState& smem_pipe_write,
         PipelineState& smem_pipe_read,
         SharedStorage &shared_storage,
         Scheduler& scheduler,
         typename Scheduler::Params const& scheduler_params,
         typename Scheduler::WorkTileInfo& work_tile_info,
         cute::tuple<int32_t, int32_t, int32_t> block_coord,
         int work_idx,
         const Seqlen_traits& seqlen_traits_q,
         const Seqlen_traits& seqlen_traits_k         
         ) {
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        // Sw<2,4,3> o smem_ptr[8b](unset) o (((_8,_8),(_16,_4)),_2,_4,_2):(((_64,_512),(_1,_16)),_4096,_8192,_32768)
        // Sw<2,4,3> o smem_ptr[8b](unset) o (((_8,_8),(_16,_4)),_1,_8,_1):(((_64,_512),(_1,_16)),_0,_4096,_0)
        using SmemLayoutTransposeV = typename Ktraits::SmemLayoutTransposeV;
        // Sw<2,4,3> o smem_ptr[8b](unset) o (((_16,_4),(_8,_8)),_2,_4,_2):(((_1,_16),(_64,_512)),_16384,_4096,_32768)
        // Sw<2,4,3> o smem_ptr[8b](unset) o (((_16,_4),(_8,_8)),_1,_8,_1):(((_1,_16),(_64,_512)),_0,_4096,_0)
        using SmemLayoutTransposeVt = typename Ktraits::SmemLayoutTransposeVt;

        auto smem_k_tensormap = shared_storage.smem_k_tensormap;
        // Sw<2,4,3>_smem_ptr[8b](0x7f6a00000400) o (_128,(_64,_9)):(_64,(_1,_8192)
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        // Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_v.smem_v.data()), SmemLayoutK{});
        // Sw<2,4,3>_smem_ptr[8b](0x7f6a00012400) o (_64,(_64,_9),_2):(_64,(_1,_4096),_36864)
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.smem_v.data()), SmemLayoutK{});
        Tensor sK = sV;
        
        // sV_divide: smem_ptr[8b](0x7f6a00012400) o Sw<2,4,3> o _0 o (((_8,_8),(_16,_4)),_1,_8,_2):(((_64,_512),(_1,_16)),_0,_4096,_32768)
        Tensor sV_divide = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_v.smem_v.data()), SmemLayoutTransposeV{}));
        // sVt_divide: smem_ptr[8b](0x7f6a00024400) o Sw<2,4,3> o _0 o (((_16,_4),(_8,_8)),_1,_8,_2):(((_1,_16),(_64,_512)),_0,_4096,_32768)
        Tensor sVt_divide = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_v.smem_v_out.data()), SmemLayoutTransposeVt{}));

        auto smem_transpose_V = SmemTransposeFp8_64x64();
        auto do_transpose_V = [&](int stage) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < shape<2>(SmemLayoutTransposeV{}); ++j) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < shape<1>(SmemLayoutTransposeV{}); ++i) {
                smem_transpose_V(flatten(sV_divide(_, i, j, stage)),
                                flatten(sVt_divide(_, i, j, stage)));
                }
            }
        };
        
        cute::TmaDescriptor* tma_load_K_desc_ptr = const_cast<cute::TmaDescriptor*>(mainloop_params.tma_load_K.get_tma_descriptor());
        bool is_page_cache = tma_load_K_page_ptr != nullptr;
        // bool is_page_cache = false;
        if (is_page_cache) {
            tma_load_K_desc_ptr = tma_load_K_page_ptr;
        }
        

        Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());
        Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.layout_K.shape()); // (s, h, n, b)
        // Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.layout_V.shape());
        // Tensor mK = mV;

        auto [m_block, bidh, bidb] = block_coord;
        int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        Tensor gQ = seqlen_traits_q.get_local_tile_tensor(
            mQ, select<0, 2>(TileShape_MNK{}), bidh, bidb)(_, _, m_block);  // (M, K)
        Tensor gK = seqlen_traits_k.get_local_tile_tensor(
            mK, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb);  // (N, K, _)
        // Tensor gV = seqlen_traits_k.get_local_tile_tensor(
        //     mV, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb);  // (N, K, _)

        Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
        Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
        auto [tQgQ, tQsQ] = tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sQ_x), group_modes<0, 2>(gQ_x));  // (TMA), (TMA)
        auto [tKgK, tKsK] = tma_partition(mainloop_params.tma_load_K, block_rank_in_cluster, Layout<ClusterShape>{},
                                          group_modes<0, 2>(sK), group_modes<0, 2>(gK));  // (TMA, k), (TMA, PIPE)
        // auto [tVgV, tVsV] = tma_partition(mainloop_params.tma_load_V, block_rank_in_cluster, Layout<ClusterShape>{},
        //                                   group_modes<0, 2>(sV), group_modes<0, 2>(gV));  // (TMA, k), (TMA, PIPE)

        uint16_t mcast_mask_kv = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int m = 0; m < size<0>(block_layout); ++m) {
                mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
            }
        }

        int page_size = mainloop_params.page_size;
        int64_t page_stride = mainloop_params.page_stride;
        int64_t row_stride = mainloop_params.row_stride;
        auto ptr_K = mainloop_params.ptr_K;

        int n_block_max = get_n_block_max(mainloop_params, m_block, seqlen_traits_q, seqlen_traits_k, is_page_cache);
        // int n_block_max = 2;
        int n_block = n_block_max - 1;

#ifdef MLA_DEBUG
        if (thread0()) {
            print("tma_load_K_desc_ptr: "); print(tma_load_K_desc_ptr); print("\n");
            print("tma_load_K_page_ptr: "); print(tma_load_K_page_ptr); print("\n");
            print("sQ: "); print(sQ); print("\n");
            print("sV: "); print(sV); print("\n");
            print("sV_divide: "); print(sV_divide); print("\n");
            print("sVt_divide: "); print(sVt_divide); print("\n");
            print("is_page_cache: "); print(is_page_cache); print("\n"); // 2
            print("n_block_max: "); print(n_block_max); print("\n"); // 2
            print("page_size: "); print(page_size); print("\n"); // 128
            print("page_stride: "); print(page_stride); print("\n"); // 73728 = 128 * 576
            print("ptr_K: "); print(ptr_K); print("\n"); // ptr[8b](0x7f1e0fed8000)
            print("tKgK: "); print(tKgK); print("\n"); // ArithTuple(_0,_0,0,1) o (((_64,_64),_9),2):(((_1@0,_1@1),_64@0),_64@1)
            print("tKsK: "); print(tKsK); print("\n"); // Sw<2,4,3>_smem_ptr[8b](0x7f1f00012400) o ((_4096,_9),_2):((_1,_4096),_36864)
            print("tQgQ: "); print(tQgQ); print("\n"); // ArithTuple(_0,0,8,2) o (((_64,_128),_9)):(((_1@0,_1@1),_64@0))
            print("tQsQ: "); print(tQsQ); print("\n"); // Sw<2,4,3>_smem_ptr[8b](0x7f1f00000400) o ((_8192,_9)):((_1,_8192))
            print("sQ_x: "); print(sQ_x); print("\n"); // Sw<2,4,3>_smem_ptr[8b](0x7f1f00000400) o ((_128,(_64,_9)),_1):((_64,(_1,_8192)),_0)
            print("gQ_x: "); print(gQ_x); print("\n"); // ArithTuple(_0,0,8,2) o ((_128,_576),_1):((_1@1,_1@0),_0)
            print("gQ: "); print(gQ); print("\n"); // ArithTuple(_0,0,8,2) o (_128,_576):(_1@1,_1@0)
            print("gK: "); print(gK); print("\n"); // ArithTuple(_0,_0,0,2) o (_64,_576,2):(_1@1,_1@0,_64@1)
            print("mQ: "); print(mQ); print("\n"); // ArithTuple(_0,_0,_0,_0) o (1,576,128,4):(_1@1,_1@0,_1@2,_1@3)
            print("mK: "); print(mK); print("\n"); // ArithTuple(_0,_0,_0,_0) o (128,576,1,4):(_1@1,_1@0,_1@2,_1@3)
        }
#endif
        int lane_predicate = cute::elect_one_sync();
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // Wait for the MMA warpgroups to say that smem_q is ready
        // for fp8, change from NumThreadsPerWarp to NumThreadsPerWarpGroup ??
        cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/); // 128 * 2 + 128
        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
            shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
            copy(mainloop_params.tma_load_Q.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_Q), 0 /*mcast_mask*/), tQgQ, tQsQ); 
        }

        shared_storage.barrier_O.wait((work_idx + 1) % 2);

        if constexpr(Is_causal) {
            if (warp_idx_in_warpgroup == 0 && lane_predicate) {
                pipeline_k.producer_acquire(smem_pipe_write);
                copy(mainloop_params.tma_load_K.with(tma_load_K_desc_ptr, *pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                    tKgK(_, n_block), tKsK(_, smem_pipe_write.index()));
                // pipeline_v.producer_acquire(smem_pipe_write);
                // copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                //     tVgV(_, n_block), tVsV(_, smem_pipe_write.index()));        
            }        
                        
            CUTLASS_PRAGMA_UNROLL
            for (int iter = 0; iter < kStages && n_block > 0; ++iter, --n_block) {
                pipeline_v.consumer_wait(smem_pipe_read);
                // pipeline_vt.producer_acquire(smem_pipe_write);
                do_transpose_V(smem_pipe_read.index());
                pipeline_vt.producer_commit(smem_pipe_write);
                pipeline_v.consumer_release(smem_pipe_read);

                ++smem_pipe_write;
                ++smem_pipe_read;
                
                if (warp_idx_in_warpgroup == 0 && lane_predicate) {
                    pipeline_k.producer_acquire(smem_pipe_write);
                    copy(mainloop_params.tma_load_K.with(tma_load_K_desc_ptr, *pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                        tKgK(_, n_block-1), tKsK(_, smem_pipe_write.index()));
                    // pipeline_v.producer_acquire(smem_pipe_write);
                    // copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                    //     tVgV(_, n_block-1), tVsV(_, smem_pipe_write.index()));
                }
            }            
            
            #pragma unroll 2
            for (; n_block > 0; --n_block) {
                pipeline_v.consumer_wait(smem_pipe_read);
                pipeline_vt.producer_acquire(smem_pipe_write);
                do_transpose_V(smem_pipe_read.index());
                pipeline_vt.producer_commit(smem_pipe_write);
                pipeline_v.consumer_release(smem_pipe_read);

                ++smem_pipe_write;
                ++smem_pipe_read;
                
                if (warp_idx_in_warpgroup == 0 && lane_predicate) {
                    pipeline_k.producer_acquire(smem_pipe_write);
                    copy(mainloop_params.tma_load_K.with(tma_load_K_desc_ptr, *pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                        tKgK(_, n_block-1), tKsK(_, smem_pipe_write.index()));
                    // pipeline_v.producer_acquire(smem_pipe_write);
                    // copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                    //     tVgV(_, n_block-1), tVsV(_, smem_pipe_write.index()));
                }                                                                
            }       

            scheduler.prefetch_next_work(scheduler_params, work_tile_info);
            scheduler.broadcast_next_work(work_tile_info);
            
            pipeline_v.consumer_wait(smem_pipe_read);
            if (n_block_max > kStages)
                pipeline_vt.producer_acquire(smem_pipe_write);
            do_transpose_V(smem_pipe_read.index());
            pipeline_vt.producer_commit(smem_pipe_write);
            pipeline_v.consumer_release(smem_pipe_read);

            ++smem_pipe_write;
            ++smem_pipe_read;
        } else {
            cute::tma_descriptor_fence_release();
            if (warp_idx_in_warpgroup == 0 && lane_predicate) {
                if (is_page_cache) {
                    int64_t global_offset = flash::resolve_page_slice_offset(block_table, n_block, kBlockN, page_size, page_stride, row_stride);
                    cute::tma_descriptor_replace_addr_in_global_mem(tma_load_K_desc_ptr, ptr_K + global_offset);
#ifdef MLA_DEBUG
                    if (thread0()) {
                        PRINT_DEBUG_SITE();
                        print("update tma_load_K_desc_ptr n_block: "); print(n_block); print("\n");
                        print("update tma_load_K_desc_ptr kBlockN: "); print(kBlockN); print("\n");
                        print("update tma_load_K_desc_ptr page_size: "); print(page_size); print("\n");
                        print("update tma_load_K_desc_ptr page_stride: "); print(page_stride); print("\n");
                        print("update tma_load_K_desc_ptr ptr_K: "); print(ptr_K); print("\n");
                        print("update tma_load_K_desc_ptr global_offset: "); print(global_offset); print("\n");
                        print("update tma_load_K_desc_ptr ptr_K + global_offset: "); print(ptr_K + global_offset); print("\n");
                    }
#endif
                }
            }
            cute::tma_descriptor_fence_acquire(tma_load_K_desc_ptr);         
            if (warp_idx_in_warpgroup == 0 && lane_predicate) {
                pipeline_k.producer_acquire(smem_pipe_write);
                copy(mainloop_params.tma_load_K.with(tma_load_K_desc_ptr, *pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                    tKgK(_, 0), tKsK(_, smem_pipe_write.index()));     
            }
            // // With fp8 kernel, smem_o is in union with smem_v_out,
            // // so could use NamedBarrier instead of ClusterBarrier.
            // // But, this doesn't appear to have any benefit.
            // shared_storage.barrier_O.wait((work_idx + 1) % 2);

            pipeline_v.consumer_wait(smem_pipe_read);
            // pipeline_vt.producer_acquire(smem_pipe_write);
            do_transpose_V(smem_pipe_read.index());
            pipeline_vt.producer_commit(smem_pipe_write);
            pipeline_v.consumer_release(smem_pipe_read);

            ++smem_pipe_write;
            ++smem_pipe_read;
            --n_block;


            constexpr int extra_iterations = kStages - 1;
            CUTLASS_PRAGMA_UNROLL
            for (int iter = 0; iter < extra_iterations && n_block >= 0; ++iter) {
                cute::tma_descriptor_fence_release();
                if (warp_idx_in_warpgroup == 0 && lane_predicate) {
                    if (is_page_cache) {
                        int64_t global_offset = flash::resolve_page_slice_offset(block_table, n_block, kBlockN, page_size, page_stride, row_stride);
                        cute::tma_descriptor_replace_addr_in_global_mem(tma_load_K_desc_ptr, ptr_K + global_offset);
#ifdef MLA_DEBUG
                        if (thread0()) {
                            PRINT_DEBUG_SITE();
                            print("update tma_load_K_desc_ptr n_block: "); print(n_block); print("\n");
                            print("update tma_load_K_desc_ptr ptr_K: "); print(ptr_K); print("\n");
                            print("update tma_load_K_desc_ptr global_offset: "); print(global_offset); print("\n");
                            print("update tma_load_K_desc_ptr ptr_K + global_offset: "); print(ptr_K + global_offset); print("\n");
                        }
#endif
                    }
                }
                cute::tma_descriptor_fence_acquire(tma_load_K_desc_ptr);          
                if (warp_idx_in_warpgroup == 0 && lane_predicate) {
                    pipeline_k.producer_acquire(smem_pipe_write);
                    copy(mainloop_params.tma_load_K.with(tma_load_K_desc_ptr, *pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                        tKgK(_, 0), tKsK(_, smem_pipe_write.index()));             
                }
                
                pipeline_v.consumer_wait(smem_pipe_read);
                // pipeline_vt.producer_acquire(smem_pipe_write);
                do_transpose_V(smem_pipe_read.index());
                pipeline_vt.producer_commit(smem_pipe_write);
                pipeline_v.consumer_release(smem_pipe_read);
                
                ++smem_pipe_write;
                ++smem_pipe_read;
                --n_block;
            }

            // CUTLASS_PRAGMA_NO_UNROLL
            // #pragma unroll 2        
            for (; n_block >= 0; --n_block) {
                cute::tma_descriptor_fence_release();
                if (warp_idx_in_warpgroup == 0 && lane_predicate) {
                    if (is_page_cache) {
                        int64_t global_offset = flash::resolve_page_slice_offset(block_table, n_block, kBlockN, page_size, page_stride, row_stride);
                        cute::tma_descriptor_replace_addr_in_global_mem(tma_load_K_desc_ptr, ptr_K + global_offset);
#ifdef MLA_DEBUG
                        if (thread0()) {
                            PRINT_DEBUG_SITE();
                            print("update tma_load_K_desc_ptr n_block: "); print(n_block); print("\n");
                            print("update tma_load_K_desc_ptr ptr_K: "); print(ptr_K); print("\n");
                            print("update tma_load_K_desc_ptr global_offset: "); print(global_offset); print("\n");
                            print("update tma_load_K_desc_ptr ptr_K + global_offset: "); print(ptr_K + global_offset); print("\n");
                        }
#endif
                    }
                }
                cute::tma_descriptor_fence_acquire(tma_load_K_desc_ptr);          
                if (warp_idx_in_warpgroup == 0 && lane_predicate) {
                    pipeline_k.producer_acquire(smem_pipe_write);
                    copy(mainloop_params.tma_load_K.with(tma_load_K_desc_ptr, *pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                        tKgK(_, 0), tKsK(_, smem_pipe_write.index()));                              
                }
                
                pipeline_v.consumer_wait(smem_pipe_read);
                pipeline_vt.producer_acquire(smem_pipe_write);
                do_transpose_V(smem_pipe_read.index());
                pipeline_vt.producer_commit(smem_pipe_write);
                pipeline_v.consumer_release(smem_pipe_read);
                
                ++smem_pipe_write;
                ++smem_pipe_read;
            }
            // scheduler.prefetch_next_work(scheduler_params, work_tile_info);
            // scheduler.broadcast_next_work(work_tile_info);
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
              PipelineState& smem_pipe_write_k, PipelineState& smem_pipe_write_v) {
        int lane_predicate = cute::elect_one_sync();
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // Issue the epilogue waits
        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
          /* This helps avoid early exit of blocks in Cluster
          * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
          * then would just be acquired since the phase was still inverted from make_producer_start_state
          */
          pipeline_k.producer_tail(smem_pipe_write_k);
          pipeline_v.producer_tail(smem_pipe_write_v);
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail_one_write(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
              PipelineState& smem_pipe_write) {
        int lane_predicate = cute::elect_one_sync();
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // Issue the epilogue waits
        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
          /* This helps avoid early exit of blocks in Cluster
          * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
          * then would just be acquired since the phase was still inverted from make_producer_start_state
          */
          pipeline_k.producer_tail(smem_pipe_write);
          pipeline_v.producer_tail(smem_pipe_write);
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_sync() {
        if constexpr (UseSchedulerBarrier) {
            cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + cutlass::canonical_warp_group_idx() /*id*/);
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_arrive() {
        if constexpr (!UseSchedulerBarrier) { return; }
        static_assert(NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup || NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
        if constexpr (NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup) {
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + (3 - cutlass::canonical_warp_group_idx()) /*id*/);
        } else {
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + (cutlass::canonical_warp_group_idx() <= 2 ? cutlass::canonical_warp_group_idx() + 1 : cutlass::canonical_warp_group_idx() + 1 - 3)  /*id*/);
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + (cutlass::canonical_warp_group_idx() <= 1 ? cutlass::canonical_warp_group_idx() + 2 : cutlass::canonical_warp_group_idx() + 2 - 3)  /*id*/);
        }
    }

    CUTLASS_DEVICE void
    mma_init() {
        // Tell producer (warp 0) that smem_q is ready
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + Ktraits::NumProducerThreads, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);                
        if constexpr (!UseSchedulerBarrier) { return; }
        static_assert(NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup || NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
        if (cutlass::canonical_warp_group_idx() > 1) {
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + 1 /*id*/);
        }
        if constexpr (NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup) {
            if (cutlass::canonical_warp_group_idx() > 2) {
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + 2 /*id*/);
            }
        }

    }

    template <typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE void
    mma(Params const& mainloop_params,
        MainloopPipeline pipeline_k,
        MainloopPipeline pipeline_v,
        PipelineState& smem_pipe_read_k,
        PipelineState& smem_pipe_read_v,
        FrgTensorO& tOrO,
        Softmax& softmax,
        int n_block_count,
        int thread_idx,
        int work_idx,
        int m_block,
        SharedStorage& shared_storage,
        const Seqlen_traits& seqlen_traits_q,
        const Seqlen_traits& seqlen_traits_k
        ) {
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
        // Tensor sK = sV;
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});

        typename Ktraits::TiledMma0 tiled_mma0;
        typename Ktraits::TiledMma1 tiled_mma1;
        auto threadMma0 = tiled_mma0.get_thread_slice(thread_idx);
        auto threadMma1 = tiled_mma1.get_thread_slice(thread_idx);

        // Allocate "fragments/descriptors" for first matmul.
        Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
        Tensor tSrK = threadMma0.partition_fragment_B(sK);
        // Allocate "fragments/descriptors" for second matmul.
        // Note: S becomes P.
        Tensor tOrV = threadMma1.partition_fragment_B(sVt);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        tiled_mma1.accumulate_ = GMMA::ScaleOut::Zero;
        int const seqlen_q = seqlen_traits_q.actual_seq_len;
        int const seqlen_k = seqlen_traits_k.actual_seq_len;
        int n_block = n_block_count - 1;

        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.barrier_Q.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.barrier_Q.wait(work_idx % 2); }

        Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{})); // (M, N)
        consumer_wait(pipeline_k, smem_pipe_read_k);
        warp_scheduler_barrier_sync();
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
        warp_scheduler_barrier_arrive();
    
        if (work_idx != 0) {
            int lane_predicate = cute::elect_one_sync();
            if (cutlass::canonical_warp_idx_sync() == Ktraits::kNWarps - 1 && lane_predicate) {
                tma_store_wait<0>();
                #pragma unroll
                for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                    shared_storage.barrier_O.arrive(cta_id, lane_predicate);
                }
            }
        }
        warpgroup_wait<0>();
        pipeline_k.consumer_release(smem_pipe_read_k); // PipelineState
        ++smem_pipe_read_k;

        auto col_limit_causal = [&](int row, int n_block) {
            return row + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
        };
        {
            Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
            Tensor tScS = threadMma0.partition_C(cS);
            #pragma unroll
            for (int i = 0; i < size(tSrS); ++i) {
                if constexpr (!Is_causal) {  // Just masking based on col
                    if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) { tSrS(i) = -INFINITY; }
                } else {  // mask based on both row and col
                    // using std::min is faster than doing col >= limit0 or col >= limit1
                    // Need to cast get<1>(tScS(i)) to (signed) int since by default it's unsigned, and the
                    // right hand side can be negative and might be converted to a very large unsigned integer.
                    if (int(get<1>(tScS(i))) >= std::min(seqlen_k - n_block * kBlockN,
                                                        col_limit_causal(int(get<0>(tScS(i))), n_block))) {
                        tSrS(i) = -INFINITY;
                    }
                }
            }
        }

        softmax.template online_softmax</*Is_first=*/true>(tSrS, mainloop_params.softmax_scale_log2);
        Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout()));
        Tensor scores_scale = make_fragment_like(softmax.row_max);
        clear(scores_scale);

        constexpr int n_masking_steps = !Is_causal ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1;
        // Only go through these if Is_causal, since n_masking_steps = 1 when !Is_causal
        #pragma unroll
        for (int masking_step = 0; masking_step < n_masking_steps - 1 && n_block > 0; ++masking_step, --n_block) {
            consumer_wait(pipeline_k, smem_pipe_read_k);
            warp_scheduler_barrier_sync();
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
            if (masking_step > 0) { softmax.rescale_o(tOrO, scores_scale); }
            consumer_wait(pipeline_v, smem_pipe_read_v);
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
            warp_scheduler_barrier_arrive();
            warpgroup_wait<1>();
            pipeline_k.consumer_release(smem_pipe_read_k);  // release K
            Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
            Tensor tScS = threadMma0.partition_C(cS);
            #pragma unroll
            for (int i = 0; i < size(tSrS); ++i) {
                if (int(get<1>(tScS(i))) >= col_limit_causal(int(get<0>(tScS(i))), n_block - 1)) {
                    tSrS(i) = -INFINITY;
                }
            }
            cute::copy(softmax.template max</*Is_first=*/false, /*Check_inf=*/true>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
            softmax.template online_softmax</*Is_first=*/false, /*Check_inf=*/true>(tSrS, mainloop_params.softmax_scale_log2);
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read_v);  // release V
            ++smem_pipe_read_k;
            ++smem_pipe_read_v;
            cute::copy(make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout())), tOrP);
        }

        #pragma unroll 1
        for (; n_block > 0; --n_block) {
            consumer_wait(pipeline_k, smem_pipe_read_k);
            warp_scheduler_barrier_sync();
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
            softmax.rescale_o(tOrO, scores_scale);
            consumer_wait(pipeline_v, smem_pipe_read_v);
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
            warp_scheduler_barrier_arrive();
            warpgroup_wait<1>();
            pipeline_k.consumer_release(smem_pipe_read_k);  // release K
            // auto scores_scale = softmax.template max</*Is_first=*/false>(tSrS);
            cute::copy(softmax.template max</*Is_first=*/false>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
            softmax.template online_softmax</*Is_first=*/false>(tSrS, mainloop_params.softmax_scale_log2);
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read_v);  // release V
            ++smem_pipe_read_k;
            ++smem_pipe_read_v;
            // softmax.rescale_o(tOrO, scores_scale);
            cute::copy(make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout())), tOrP);
        }
        // Tell warp 0 that smem_q is ready
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);
        softmax.rescale_o(tOrO, scores_scale);
        consumer_wait(pipeline_v, smem_pipe_read_v);
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
        cute::copy(softmax.template finalize</*Check_inf=*/Is_causal>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
        warpgroup_wait<0>();
        pipeline_v.consumer_release(smem_pipe_read_v);  // release V, otherwise producers will hang
        ++smem_pipe_read_v;

        softmax.rescale_o(tOrO, scores_scale);
        return;
    }

    template <bool Delay_V_release = false, typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE void
    mma_fp8(Params const& mainloop_params,
        MainloopPipeline pipeline_k,
        MainloopPipelineNoTMA pipeline_vt,
        PipelineState& smem_pipe_read,
        PipelineState& smem_pipe_release,        
        FrgTensorO& tOrO,
        Softmax& softmax,
        int n_block_count,
        int thread_idx,
        int work_idx,
        int m_block,
        SharedStorage& shared_storage,
        const Seqlen_traits& seqlen_traits_q,
        const Seqlen_traits& seqlen_traits_k
        ) {
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_v.smem_v.data()), SmemLayoutK{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.smem_v_out.data()), SmemLayoutVt{});

        typename Ktraits::TiledMma0 tiled_mma0;
        typename Ktraits::TiledMma1 tiled_mma1;
        auto threadMma0 = tiled_mma0.get_thread_slice(thread_idx);
        auto threadMma1 = tiled_mma1.get_thread_slice(thread_idx);

        // Allocate "fragments/descriptors" for first matmul.
        Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
        Tensor tSrK = threadMma0.partition_fragment_B(sK);
        // Allocate "fragments/descriptors" for second matmul.
        Tensor tOrV = threadMma1.partition_fragment_B(sVt);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        tiled_mma1.accumulate_ = GMMA::ScaleOut::Zero;
        // workaround for fp8 only perf regression pending change to seqlen traits class
        int const seqlen_q = Seqlen_traits::kUseVarSeqLen ? seqlen_traits_q.actual_seq_len : shape<0>(mainloop_params.layout_Q);
        int const seqlen_k = Seqlen_traits::kUseVarSeqLen ? seqlen_traits_k.actual_seq_len : shape<0>(mainloop_params.layout_K);
        int n_block = n_block_count - 1;
        
        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.barrier_Q.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.barrier_Q.wait(work_idx % 2); }
        
        Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));        
        
        // save complete.
        if (work_idx != 0) {        
            int lane_predicate = cute::elect_one_sync();
            if (cutlass::canonical_warp_idx_sync() == Ktraits::kNWarps - 1 && lane_predicate) {
                tma_store_wait<0>();
                #pragma unroll
                for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
                    shared_storage.barrier_O.arrive(cta_id, lane_predicate);
                }
            }        
        }

        consumer_wait(pipeline_k, smem_pipe_read);                        
        warp_scheduler_barrier_sync();
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
        warpgroup_wait<0>();
        warp_scheduler_barrier_arrive();
        pipeline_k.consumer_release(smem_pipe_read);

        auto col_limit_causal = [&](int row, int n_block) {
            return row + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
        };       
        {
            Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
            Tensor tScS = threadMma0.partition_C(cS);
            #pragma unroll
            for (int i = 0; i < size(tSrS); ++i) {
                if constexpr (!Is_causal) {  // Just masking based on col                
                    if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) { tSrS(i) = -INFINITY; }
                } else {  // mask based on both row and col
                    if (int(get<1>(tScS(i))) >= std::min(seqlen_k - n_block * kBlockN,
                                                         col_limit_causal(int(get<0>(tScS(i))), n_block))) {
                        tSrS(i) = -INFINITY;
                    }
                }
            }
        }

        softmax.template online_softmax</*Is_first=*/true>(tSrS, mainloop_params.softmax_scale_log2);
        Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
        permute_regs_A_to_C(tOrP);
        
        Tensor scores_scale = make_fragment_like(softmax.row_max);
        clear(scores_scale);
        
        consumer_wait(pipeline_vt, smem_pipe_read);
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);                
        if constexpr(!Delay_V_release) { pipeline_vt.consumer_release(smem_pipe_read); }

        ++smem_pipe_read;
        --n_block;
        constexpr int extra_iterations = !Is_causal ? kStages - 1 : cute::ceil_div(kBlockM, kBlockN);        

        if constexpr(Is_causal) {
            CUTLASS_PRAGMA_UNROLL      
            for (int iter = 0; iter < extra_iterations && n_block >= 0; ++iter, --n_block) {
                Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
                consumer_wait(pipeline_k, smem_pipe_read);
                warp_scheduler_barrier_sync();
                flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);

                Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
                Tensor tScS = threadMma0.partition_C(cS);
                #pragma unroll
                for (int i = 0; i < size(tSrS); ++i) {
                    if (int(get<1>(tScS(i))) >= col_limit_causal(int(get<0>(tScS(i))), n_block)) {
                        tSrS(i) = -INFINITY;
                    }
                }

                warp_scheduler_barrier_arrive();
                pipeline_k.consumer_release(smem_pipe_read);
                consumer_wait(pipeline_vt, smem_pipe_read);
                
                cute::copy(softmax.template max</*Is_first=*/false, /*Check_inf=*/true>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
                softmax.rescale_o(tOrO, scores_scale);
                softmax.template online_softmax</*Is_first=*/false, /*Check_inf=*/true>(tSrS, mainloop_params.softmax_scale_log2);
                Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
                permute_regs_A_to_C(tOrP);

                if constexpr(Delay_V_release) {
                    pipeline_vt.consumer_release(smem_pipe_release);
                    ++smem_pipe_release;
                }
                flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);            
                if constexpr(!Delay_V_release) { pipeline_vt.consumer_release(smem_pipe_read); }                
                ++smem_pipe_read;
            }
        } else {
            CUTLASS_PRAGMA_UNROLL      
            for (int iter = 0; iter < extra_iterations && n_block >= 0; ++iter, --n_block) {
                Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
                consumer_wait(pipeline_k, smem_pipe_read);
                if constexpr(Delay_V_release) {
                    pipeline_vt.consumer_release(smem_pipe_release);
                    ++smem_pipe_release;
                }
                warp_scheduler_barrier_sync();
                flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                warp_scheduler_barrier_arrive();
                if constexpr(!Delay_V_release) { pipeline_k.consumer_release(smem_pipe_read); }
                else { consumer_wait(pipeline_vt, smem_pipe_read); }
                
                cute::copy(softmax.template max</*Is_first=*/false>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
                softmax.rescale_o(tOrO, scores_scale);
                softmax.template online_softmax</*Is_first=*/false>(tSrS, mainloop_params.softmax_scale_log2);
                Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
                permute_regs_A_to_C(tOrP);

                if constexpr (Delay_V_release) { pipeline_k.consumer_release(smem_pipe_read); }
                else { consumer_wait(pipeline_vt, smem_pipe_read); }
                flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                if constexpr(!Delay_V_release) { pipeline_vt.consumer_release(smem_pipe_read); }                
                ++smem_pipe_read;
            }
        }

        if constexpr(Delay_V_release) {
            warp_scheduler_barrier_sync();
            CUTLASS_PRAGMA_NO_UNROLL
            for (; n_block >= 0; --n_block) {
                Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
                consumer_wait(pipeline_k, smem_pipe_read);
                pipeline_vt.consumer_release(smem_pipe_release);                
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                warp_scheduler_barrier_arrive();
                warpgroup_wait<0>();                
                consumer_wait(pipeline_vt, smem_pipe_read);

                cute::copy(softmax.template max</*Is_first=*/false>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
                softmax.rescale_o(tOrO, scores_scale);
                softmax.template online_softmax</*Is_first=*/false>(tSrS, mainloop_params.softmax_scale_log2);
                Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
                permute_regs_A_to_C(tOrP);

                pipeline_k.consumer_release(smem_pipe_read);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                warp_scheduler_barrier_sync();
                warpgroup_wait<0>();
                ++smem_pipe_read;
                ++smem_pipe_release;
            }
            warp_scheduler_barrier_arrive();
            pipeline_vt.consumer_release(smem_pipe_release);
            ++smem_pipe_release;
        } else {
            if constexpr (kHeadDim == 128) { warp_scheduler_barrier_sync(); }
            CUTLASS_PRAGMA_NO_UNROLL
            for (; n_block >= 0; --n_block) {
                Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
                consumer_wait(pipeline_k, smem_pipe_read);
                if constexpr (kHeadDim == 256) { warp_scheduler_barrier_sync(); }
                flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                warp_scheduler_barrier_arrive();
                pipeline_k.consumer_release(smem_pipe_read);

                cute::copy(softmax.template max</*Is_first=*/false>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
                softmax.rescale_o(tOrO, scores_scale);
                softmax.template online_softmax</*Is_first=*/false>(tSrS, mainloop_params.softmax_scale_log2);
                Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
                permute_regs_A_to_C(tOrP);

                consumer_wait(pipeline_vt, smem_pipe_read);
                if constexpr (kHeadDim == 128) { warp_scheduler_barrier_sync(); }
                flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                pipeline_vt.consumer_release(smem_pipe_read);
                ++smem_pipe_read;
            }
            if constexpr (kHeadDim == 128) { warp_scheduler_barrier_arrive(); }
        }
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::QueryEmpty) /*id*/);
        
        cute::copy(softmax.template finalize</*Check_inf=*/Is_causal>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
        softmax.rescale_o(tOrO, scores_scale);
        return;
    }

};

} // namespace flash
