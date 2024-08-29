#include <fstream>
#include <torch/torch.h>
#include "flash_api.cpp"

int main(int argc, const char *argv[]) {
    std::cout << "hello world" << std::endl;
    int batch_size = 4;
    int seqlen_q = 1;
    // int seqlen = 128;
    int num_heads = 128;
    // int num_heads_k = 1;
    // int head_size = 256;
    int head_size = 576;
    int out_head_size = 512;
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kHalf);
    at::Tensor q = torch::randn({batch_size, seqlen_q, num_heads, head_size}, options).to(torch::kFloat8_e4m3fn);

    // page table
    int num_blocks = batch_size;
    int page_size = 64;
    at::Tensor cache = torch::randn({num_blocks, page_size, 1, head_size}, options).to(torch::kFloat8_e4m3fn);

    // at::Tensor k = torch::randn({batch_size * seqlen, num_heads_k, head_size}, options).to(torch::kFloat8_e4m3fn);
    // at::Tensor v = k;
    at::Tensor o = torch::zeros({batch_size, seqlen_q, num_heads, out_head_size},  options);
    // std::cout << q.sizes() << " " << q.device().type() << " " << q.layout() << std::endl;

    // // Prepare the optional tensors for output and alibi slopes
    // c10::optional<at::Tensor> out_ = c10::nullopt;;
    // c10::optional<at::Tensor> alibi_slopes_ = c10::nullopt;  // Assuming alibi_slopes are not used here

    // // Prepare additional required arguments
    // float p_dropout = 0.0f;  // Example dropout probability
    // float softmax_scale = 1.0f;  // Example scale for softmax
    // bool is_causal = true;
    // int window_size_left = -1;  // Example window size
    // int window_size_right = -1;  // Example window size
    // bool return_softmax = false;  // Assuming we don't need to return softmax scores
    // c10::optional<at::Generator> gen_ = c10::nullopt;  // Assuming no specific generator needed

    // auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(q.device());

    // auto cu_seqlens_q = torch::arange(0, (batch_size + 1) * seqlen_q, seqlen_q, int_options);
    // auto cu_seqlens_k = torch::arange(0, (batch_size + 1) * seqlen, seqlen, int_options);
    // c10::optional<at::Tensor> seqused_k = c10::nullopt;
    // int max_seqlen_q = seqlen_q;
    // int max_seqlen_k = seqlen;

    // std::vector<at::Tensor> result = mha_varlen_fwd(q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k, max_seqlen_q, seqlen, softmax_scale, is_causal);

    // std::vector<at::Tensor>
    // mla_fwd(at::Tensor &q,   // batch_size x 1 x num_heads x head_size
    //         const at::Tensor &cache,  // num_blocks x page_block_size x 1 x head_size
    //         c10::optional<at::Tensor> &out_, // batch_size x 1 x num_heads x head_size
    //         at::Tensor &block_table, // batch_size x max_num_blocks_per_seq
    //         const float softmax_scale) {

    c10::optional<at::Tensor> out_ = o;
    int max_num_page_per_seq = 10;
    at::Tensor block_table = torch::randint(0, num_blocks - 1, {batch_size, max_num_page_per_seq}, options.dtype(torch::kInt32));
    at::Tensor seqlens = torch::tensor({256, 256, 256, 256}, options.dtype(torch::kInt32));

    // auto q_view = q.view({batch_size, num_heads, seqlen_q, head_size});
    std::vector<at::Tensor> result = mla_kvcache_fwd(q, cache, seqlens, block_table, out_, 1.0f);
    std::cout << result[0].sizes() << std::endl;

    std::cout << "q: " << q.sizes() << std::endl;
    std::cout << "cache: " << cache.sizes() << std::endl;
    std::cout << "seqlens: " << seqlens.sizes() << std::endl;
    std::cout << "block_table: " << block_table.sizes() << std::endl;
    std::cout << "Done." << std::endl;

    std::cout << o << std::endl;

    return 0;
}