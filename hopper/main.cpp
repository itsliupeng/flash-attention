#include <fstream>
#include <torch/torch.h>
#include "flash_api.cpp"

int main(int argc, const char *argv[]) {
    std::cout << "hello world" << std::endl;
    int batch_size = 8;
    int seqlen_q = 128;
    int seqlen = 128;
    int num_heads = 16;
    int num_heads_k = 1;
    // int head_size = 256;
    int head_size = 576;
    int out_head_size = 512;
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kHalf);
    at::Tensor q = torch::randn({batch_size * seqlen_q, num_heads, head_size}, options).to(torch::kFloat8_e4m3fn);
    at::Tensor k = torch::randn({batch_size * seqlen, num_heads_k, head_size}, options).to(torch::kFloat8_e4m3fn);
    at::Tensor v = k;
    // at::Tensor o = torch::randn({batch_size, seqlen_q, num_heads, out_head_size},  options);
    std::cout << q.sizes() << " " << q.device().type() << " " << q.layout() << std::endl;

    // Prepare the optional tensors for output and alibi slopes
    c10::optional<at::Tensor> out_ = c10::nullopt;;
    c10::optional<at::Tensor> alibi_slopes_ = c10::nullopt;  // Assuming alibi_slopes are not used here

    // Prepare additional required arguments
    float p_dropout = 0.0f;  // Example dropout probability
    float softmax_scale = 1.0f;  // Example scale for softmax
    bool is_causal = true;
    int window_size_left = -1;  // Example window size
    int window_size_right = -1;  // Example window size
    bool return_softmax = false;  // Assuming we don't need to return softmax scores
    c10::optional<at::Generator> gen_ = c10::nullopt;  // Assuming no specific generator needed

    // std::vector<at::Tensor> result = mha_fwd(q, k, v, out_, softmax_scale, is_causal);

    // mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    //             const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    //             const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    //             c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    //             const at::Tensor &cu_seqlens_q,  // b+1
    //             const at::Tensor &cu_seqlens_k,  // b+1
    //             c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
    //             int max_seqlen_q,
    //             const int max_seqlen_k,
    //             const float softmax_scale,
    //             bool is_causal) {

    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(q.device());

    auto cu_seqlens_q = torch::arange(0, (batch_size + 1) * seqlen_q, seqlen_q, int_options);
    auto cu_seqlens_k = torch::arange(0, (batch_size + 1) * seqlen, seqlen, int_options);
    c10::optional<at::Tensor> seqused_k = c10::nullopt;
    int max_seqlen_q = seqlen_q;
    int max_seqlen_k = seqlen;

    std::vector<at::Tensor> result = mha_varlen_fwd(q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k, max_seqlen_q, seqlen, softmax_scale, is_causal);

    // std::cout << out_.value().sizes() << std::endl;
    std::cout << result[0].sizes() << std::endl;

    std::cout << "Done." << std::endl;

    return 0;
}