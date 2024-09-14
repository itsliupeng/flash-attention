#include <torch/torch.h>
#include <iostream>
#include <cutlass/cutlass.h>
#include "cute/tensor.hpp"

int main() {

    using namespace cute;
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kHalf);
    int B = 8;
    int page_size = 4;
    int H = 32;
    // 创建一个递增的张量，范围是从 0 开始到 4 * 64 * 1 * 576
    auto data = torch::arange(B * page_size * 1 * H, torch::kFloat32);
    auto max_val = torch::max(data);
    // 将张量 reshape 为 [4, 64, 1, 576] 的形状
    auto cache = (data.view({B, page_size, 1, H}) / max_val * 100).to(options.dtype(torch::kFloat8_e4m3fn));

    int max_num_page_per_seq = 4;
    auto block_table = torch::randint(0, B, {1, max_num_page_per_seq}, options.dtype(torch::kInt32));

    // 打印张量
    std::cout << block_table << std::endl;
    std::cout << cache.sizes() << std::endl;
    // std::cout << cache << std::endl;

    Tensor cute_tensor = make_tensor(make_gmem_ptr(cache.data_ptr()), make_layout(make_shape(H, 1, page_size, B)));
    print(cute_tensor);
    print(make_layout(make_shape(H, 1, page_size, B)));

    return 0;
}
