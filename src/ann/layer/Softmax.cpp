/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this template
 */

/*
 * File:   Softmax.cpp
 * Author: ltsach
 *
 * Created on August 25, 2024, 2:46 PM
 */

#include "layer/Softmax.h"
#include "ann/functions.h"
#include "sformat/fmt_lib.h"
#include <filesystem> //require C++17
namespace fs = std::filesystem;

Softmax::Softmax(int axis, string name) : m_nAxis(axis) {
    if(trim(name).size() != 0) m_sName = name;
    else m_sName = "Softmax_" + to_string(++m_unLayer_idx);
}

Softmax::Softmax(const Softmax &orig) {}

Softmax::~Softmax() {}

xt::xarray<double> Softmax::forward(xt::xarray<double> X) {
    // Tạo một xarray để lưu kết quả softmax
    xt::xarray<double> m_aCached_Y(X.shape());

    // Tính giá trị max trong mỗi hàng để tránh tràn số
    auto max_values = xt::amax(X, 1); // Tìm giá trị max của mỗi dòng (axis 1)

    // Lặp qua từng hàng trong ma trận X
    for(std::size_t i = 0; i < X.shape()[0]; ++i) {
        double sum_exp = 0.0;

        // Tính tổng của e^(z_i - max_value) cho mỗi phần tử z_i trong dòng
        for(std::size_t j = 0; j < X.shape()[1]; ++j) {
            sum_exp += std::exp(X(i, j) - max_values(i));
        }

        // Lặp lại một lần nữa để tính giá trị softmax
        for(std::size_t j = 0; j < X.shape()[1]; ++j) {
            m_aCached_Y(i, j) = std::exp(X(i, j) - max_values(i)) / sum_exp;
        }
    }

    // Lưu kết quả softmax vào m_aCached_Y
    return m_aCached_Y;
}

xt::xarray<double> Softmax::backward(xt::xarray<double> DY) {
    xt::xarray<double> m_aCached_Y_1D = xt::squeeze(m_aCached_Y);

    // Tạo ma trận chéo DIAG(y)
    xt::xarray<double> diag_y = xt::diag(m_aCached_Y_1D);

    // Tính outer product y * y^T
    xt::xarray<double> outer_product = xt::linalg::outer(m_aCached_Y_1D, m_aCached_Y_1D);

    // Tính \(\operatorname{DIAG}(\mathbf{y}) - \mathbf{y} \otimes \mathbf{y}^T\)
    xt::xarray<double> result = diag_y - outer_product;

    // Kiểm tra sự tương thích giữa kích thước của result và DY
    if(result.shape(1) != DY.shape(0)) {
        throw std::runtime_error("Shape mismatch in Softmax backward: result's number of columns does not match DY's "
                                 "number of rows. SOFTMAX BACKWARD");
    }

    // Tính gradient DX
    xt::xarray<double> DX = xt::linalg::dot(result, DY);

    return DX;
}

string Softmax::get_desc() {
    string desc = fmt::format("{:<10s}, {:<15s}: {:4d}", "Softmax", this->getname(), m_nAxis);
    return desc;
}
