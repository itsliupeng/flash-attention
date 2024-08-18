#include <fstream>
#include <torch/torch.h>
#include "flash_api.cpp"

int main(int argc, const char *argv[]) {
    std::cout << "hello world" << std::endl;
    int batch_size = 8;
    int seqlen_q = 128;
    int num_heads = 16;
    // int head_size = 256;
    int head_size = 576;
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kHalf);
    // auto options = torch::TensorOptions().device(torch::kCUDA).dtype(at::ScalarType::Float8_e4m3fn);
    at::Tensor q = torch::randn({batch_size, seqlen_q, num_heads, head_size},  options).to(torch::kFloat8_e4m3fn);
    at::Tensor k = torch::randn({batch_size, seqlen_q, num_heads, head_size},  options).to(torch::kFloat8_e4m3fn);
    at::Tensor v = torch::randn({batch_size, seqlen_q, num_heads, head_size}, options).to(torch::kFloat8_e4m3fn);
    at::Tensor o = torch::randn({batch_size, seqlen_q, num_heads, head_size},  options);
    std::cout << q.sizes() << " " << q.device().type() << " " << q.layout() << std::endl;

    // Prepare the optional tensors for output and alibi slopes
    c10::optional<at::Tensor> out_ = o;
    c10::optional<at::Tensor> alibi_slopes_ = c10::nullopt;  // Assuming alibi_slopes are not used here

    // Prepare additional required arguments
    float p_dropout = 0.0f;  // Example dropout probability
    float softmax_scale = 1.0f;  // Example scale for softmax
    bool is_causal = true;
    int window_size_left = -1;  // Example window size
    int window_size_right = -1;  // Example window size
    bool return_softmax = false;  // Assuming we don't need to return softmax scores
    c10::optional<at::Generator> gen_ = c10::nullopt;  // Assuming no specific generator needed


// mha_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
//         const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
//         const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
//         c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
//         const float softmax_scale,
//         bool is_causal);

    // Call the function
    std::vector<at::Tensor> result = mha_fwd(q, k, v, out_, softmax_scale, is_causal);

    std::cout << out_.value().sizes() << std::endl;

    std::cout << "Done." << std::endl;

    return 0;
}