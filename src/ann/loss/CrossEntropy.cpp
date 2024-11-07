/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt
 * to change this license Click
 * nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this
 * template
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

CrossEntropy::CrossEntropy(const CrossEntropy& orig) : ILossLayer(orig) {}

CrossEntropy::~CrossEntropy() {}

double CrossEntropy::forward(xt::xarray<double> X, xt::xarray<double> t) {
        m_aCached_Ypred = X; 
    
        if (t.dimension() == 1) {
            xt::xarray<double> t_reshaped = xt::zeros<double>(xt::svector<size_t>{static_cast<size_t>(t.shape()[0]), 1});

            for(size_t i = 0; i < t.shape()[0]; ++i) {
                t_reshaped(i, 0) = t(i);
            }
            m_aYtarget = t_reshaped;
        } else {
            m_aYtarget = t;
        }
        const double EPSILON = 1e-7;
        if (m_aYtarget.shape(1) == 1) {
            size_t batch_size = m_aYtarget.shape(0);
            size_t num_classes = X.shape(1);
            xt::xarray<double> temp_target = xt::zeros<double>({batch_size, num_classes});
            for (size_t i = 0; i < batch_size; ++i) {
                int class_index = static_cast<int>(m_aYtarget(i, 0));
                if (class_index >= 0 && class_index < static_cast<int>(num_classes)) {
                    temp_target(i, class_index) = 1.0;
                }
            }
            m_aYtarget = temp_target;
        }

        xt::xarray<double> safe_x = xt::clip(X, EPSILON, 1.0);
        xt::xarray<double> log_probs = xt::log(safe_x);
        
        xt::xarray<double> losses = xt::sum(-m_aYtarget * log_probs, {1});
        double total_loss = xt::sum(losses)();
        size_t N_norm = X.shape(0);
 
        double final_loss;
        switch (m_eReduction) {
            case REDUCE_MEAN:
                final_loss = total_loss / N_norm;
                break;
            case REDUCE_SUM:
                final_loss = total_loss;
                break;
            default:
                final_loss = total_loss / N_norm;
        }
        return final_loss;
    }

xt::xarray<double> CrossEntropy::backward() {
        const double EPSILON = 1e-7;
        size_t N_norm = m_aCached_Ypred.shape(0);

        xt::xarray<double> denominator = xt::clip(m_aCached_Ypred, EPSILON, 1.0);
        xt::xarray<double> grad = -m_aYtarget / denominator;

        switch (m_eReduction) {
            case REDUCE_MEAN:
                grad = grad / static_cast<double>(N_norm);
                break;
            case REDUCE_SUM:
                break;
            default:
                grad = grad / static_cast<double>(N_norm);
        }
        return grad;
    }
