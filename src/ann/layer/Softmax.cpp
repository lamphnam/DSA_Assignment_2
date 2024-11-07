/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt
 * to change this license Click
 * nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this
 * template
 */

/*
 * File:   Softmax.cpp
 * Author: ltsach
 *
 * Created on August 25, 2024, 2:46 PM
 */

#include "layer/Softmax.h"

#include <filesystem>  //require C++17

#include "ann/functions.h"
#include "sformat/fmt_lib.h"
namespace fs = std::filesystem;

Softmax::Softmax(int axis, string name) : m_nAxis(axis) {
  if (trim(name).size() != 0)
    m_sName = name;
  else
    m_sName = "Softmax_" + to_string(++m_unLayer_idx);
}

Softmax::Softmax(const Softmax& orig) {}

Softmax::~Softmax() {}

xt::xarray<double> Softmax::forward(xt::xarray<double> X) {
    // Gọi hàm softmax từ functions.h với tham số axis được chỉ định
    xt::xarray<double> Y = softmax(X, m_nAxis);
    
    // Cache kết quả để sử dụng trong backward pass
    this->m_aCached_Y = Y;
    
    return Y;
}

xt::xarray<double> Softmax::backward(xt::xarray<double> DY) {
    xt::xarray<double> Y = this->m_aCached_Y;
    
    // Helper function to compute single sample gradient
    auto compute_single_gradient = [](const xt::xarray<double>& y, const xt::xarray<double>& dy) {
        // Create diagonal matrix from y
        auto diag_matrix = xt::diag(y);
        // Compute outer product
        auto outer_product = xt::linalg::outer(y, y);
        // Compute Jacobian
        auto jacobian = diag_matrix - outer_product;
        // Compute gradient
        return xt::linalg::dot(jacobian, dy);
    };

    // Get batch size and feature dimension
    size_t batch_size = Y.shape()[0];
    size_t feature_dim = Y.shape()[1];

    // Initialize output gradient array
    xt::xarray<double> gradient = xt::zeros<double>({batch_size, feature_dim});

    // Special case for single sample
    if (batch_size == 1) {
        auto single_grad = compute_single_gradient(
            xt::view(Y, 0),
            xt::view(DY, 0)
        );
        xt::view(gradient, 0) = single_grad;
        return gradient;
    }

    // Batch processing
    #pragma omp parallel for if(batch_size > 100)
    for (size_t i = 0; i < batch_size; ++i) {
        // Extract current sample
        auto current_y = xt::view(Y, i);
        auto current_dy = xt::view(DY, i);
        
        // Compute gradient for current sample
        auto current_grad = compute_single_gradient(current_y, current_dy);
        
        // Store result
        xt::view(gradient, i) = current_grad;
    }

    return gradient;
}
string Softmax::get_desc() {
  string desc = fmt::format("{:<10s}, {:<15s}: {:4d}", "Softmax",
                            this->getname(), m_nAxis);
  return desc;
}
