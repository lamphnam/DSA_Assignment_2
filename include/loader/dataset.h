/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt
 * to change this license Click
 * nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this
 * template
 */

/*
 * File:   dataset.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 3:59 PM
 */

#ifndef DATASET_H
#define DATASET_H
#include "ann/xtensor_lib.h"

using namespace std;

template <typename DType, typename LType>
class DataLabel {
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;

public:
    DataLabel(xt::xarray<DType> data, xt::xarray<LType> label) : data(data), label(label) {}

    xt::xarray<DType> getData() const {
        return data;
    }

    xt::xarray<LType> getLabel() const {
        return label;
    }
};

template <typename DType, typename LType>
class Batch {
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;

public:
    Batch(xt::xarray<DType> data, xt::xarray<LType> label) : data(data), label(label) {}

    Batch() {} // Default

    virtual ~Batch() {}

    xt::xarray<DType> &getData() {
        return data;
    }

    xt::xarray<LType> &getLabel() {
        return label;
    }
};

template <typename DType, typename LType>
class Dataset {
private:
public:
    Dataset() {};
    virtual ~Dataset() {};

    virtual int len() = 0;
    virtual DataLabel<DType, LType> getitem(int index) = 0;
    virtual xt::svector<unsigned long> get_data_shape() = 0;
    virtual xt::svector<unsigned long> get_label_shape() = 0;
};

//////////////////////////////////////////////////////////////////////
template <typename DType, typename LType>
class TensorDataset : public Dataset<DType, LType> {
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
    xt::svector<unsigned long> data_shape, label_shape;

public:
    /* TensorDataset:
     * need to initialize:
     * 1. data, label;
     * 2. data_shape, label_shape
     */
    TensorDataset(xt::xarray<DType> data, xt::xarray<LType> label) : data(data), label(label) {
        /* TODO: your code is here for the initialization
         */
        data_shape = data.shape();
        if(label.size() > 0) {
            label_shape = label.shape();
        } else {
            label_shape = {0};
        }
    }

    /* len():
     *  return the size of dimension 0
     */
    int len() override {
        /* TODO: your code is here to return the dataset's length
         */
        return this->data.shape()[0];
    }

    /* getitem:
     * return the data item (of type: DataLabel) that is specified by index
     */
    DataLabel<DType, LType> getitem(int index) override {
        if(index < 0 || index >= data.shape(0)) {
            throw std::out_of_range("Index is out of range!");
        }
        try {
            xt::xarray<DType> sample_data = xt::view(this->data, index, xt::all());
            xt::xarray<LType> sample_label;
            if(this->label.shape().size() > 0) {
                // If labels exist, extract the corresponding row/slice
                sample_label = xt::view(this->label, index, xt::all());
            } else {
                sample_label = this->label;
            }
            return DataLabel<DType, LType>(sample_data, sample_label);
        } catch(const std::bad_array_new_length &e) {
            return DataLabel<DType, LType>(this->data[index], this->label[index]);
        }
    }

    xt::svector<unsigned long> get_data_shape() {
        /* TODO: your code is here to return data_shape
         */
        return this->data_shape;
    }

    xt::svector<unsigned long> get_label_shape() {
        /* TODO: your code is here to return label_shape
         */
        return this->label_shape;
    }
};

#endif /* DATASET_H */
