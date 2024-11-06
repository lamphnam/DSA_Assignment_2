/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this template
 */

/*
 * File:   CrossEntropy.cpp
 * Author: ltsach
 *
 * Created on August 25, 2024, 2:47 PM
 */

#include "loss/CrossEntropy.h"
#include "ann/functions.h"

CrossEntropy::CrossEntropy(LossReduction reduction) : ILossLayer(reduction) {}

CrossEntropy::CrossEntropy(const CrossEntropy &orig) : ILossLayer(orig) {}

CrossEntropy::~CrossEntropy() {}

double CrossEntropy::forward(xt::xarray<double> X, xt::xarray<double> t) {
    // YOUR CODE IS HERE
    // Dung cho backward:3
    m_aCached_Ypred = X;
    m_aYtarget = t;
    //
    auto N = X.shape(0);
    double loss = 0.0;
    double Nnorm;
    // So sanh reduction tim N
    switch(m_eReduction) {
        case LossReduction::REDUCE_MEAN:
            Nnorm = N;
            break;
        case LossReduction::REDUCE_SUM:
        case LossReduction::REDUCE_NONE:
            Nnorm = 1.0;
            break;
    }

    if(t.dimension() == 2) {
        auto log_X = xt::log(X);

        auto elementwise_product = t * log_X;

        double sum = xt::sum(elementwise_product)();

        loss = -sum / N;
    } else if(t.dimension() == 1) {
        xt::xarray<double> log_probs = xt::log(X);

        double sum_result = 0.0;
        for(size_t i = 0; i < N; i++) {
            sum_result += log_probs(i, t(i));
        }
        loss = -sum_result / N;
    }
    return loss;
}

xt::xarray<double> CrossEntropy::backward() {
    const double EPSILON = 1e-7;
    double Nnorm;

    // Kiểm tra reduction mode bằng enum
    switch(m_eReduction) {
        case LossReduction::REDUCE_MEAN:
            Nnorm = m_aCached_Ypred.shape(0);
            break;
        case LossReduction::REDUCE_SUM:
        case LossReduction::REDUCE_NONE:
            Nnorm = 1.0;
            break;
    }

    // Áp dụng công thức với Nnorm tương ứng
    auto dY = (-m_aYtarget / (m_aCached_Ypred + EPSILON)) / Nnorm;

    return dY;
}