/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt
 * to change this license Click
 * nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this
 * template
 */

/*
 * File:   dataloader.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 4:01 PM
 */

#ifndef DATALOADER_H
#define DATALOADER_H
#include "ann/dataset.h"
#include "ann/xtensor_lib.h"
#include "list/listheader.h"

using namespace std;
// template<typename DType, typename LType>

template < typename DType, typename LType >
class DataLoader {
public:
    class Iterator {
    public:
        Iterator(DataLoader &data_loader, int batch_index) : data_loader(data_loader), batch_index(batch_index) {}

        Batch< DType, LType > operator*() { return data_loader.get_batch(batch_index); }

        Iterator &operator++() {
            batch_index++;
            return *this;
        }
        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator!=(const Iterator &other) const { return batch_index != other.batch_index; }

    private:
        DataLoader &data_loader;
        int batch_index;
    };

private:
    Dataset< DType, LType > *ptr_dataset;
    int batch_size;
    bool shuffle;
    bool drop_last;
    int dataset_size;
    int m_seed;
    xt::xarray< unsigned long > indices;
    int total_batches;

public:
    DataLoader(Dataset< DType, LType > *ptr_dataset, int batch_size, bool shuffle = true, bool drop_last = false,
               int seed = -1) {
        /*TODO: Add your code to do the initialization */
        this->ptr_dataset = ptr_dataset;
        this->batch_size = batch_size;
        this->shuffle = shuffle;
        this->drop_last = drop_last;
        this->m_seed = seed;
        dataset_size = ptr_dataset->len();
        if (batch_size > dataset_size) {
            total_batches = 0;
        } else {
            total_batches = dataset_size / batch_size;
        }

        // Resize and initialize indices vector
        indices.resize({static_cast< size_t >(dataset_size)});
        for (int i = 0; i < dataset_size; i++) {
            indices[i] = i;
        }

        // Shuffle

        if (shuffle) {

            xt::random::default_engine_type engine;
            if (m_seed >= 0) {
                xt::random::seed(m_seed);
                engine.seed(m_seed);
            }
            xt::random::shuffle(indices, engine);
        }
    }

    // Get Batch

    Batch< DType, LType > get_batch(int batch_index) {
        int start_index = batch_index * batch_size;
        int end_index = std::min((batch_index + 1) * batch_size, dataset_size);

        if (!drop_last && batch_index == total_batches - 1) {
            end_index = dataset_size;
        }

        int current_batch_size = end_index - start_index;

        if (current_batch_size <= 0) {
            throw std::runtime_error("Invalid batch size");
        }
        // Get shapes from the dataset
        auto batch_data_shape = this->ptr_dataset->get_data_shape();
        batch_data_shape[0] = current_batch_size;  // Update batch size in data shape
        auto batch_label_shape = this->ptr_dataset->get_label_shape();
        batch_label_shape[0] = current_batch_size;  // Update batch size in label shape

        // Create xarrays for batch data and labels
        xt::xarray< DType > batch_data = xt::empty< DType >(batch_data_shape);
        xt::xarray< LType > batch_label = xt::empty< LType >(batch_label_shape);

        // Populate the batch data and labels
        for (int i = start_index; i < end_index; ++i) {
            // try {
            int index = indices[i];

            DataLabel< DType, LType > data_label = ptr_dataset->getitem(index);

            auto data = data_label.getData();
            auto label = data_label.getLabel();

            // Ensure we are using the correct index to populate the batch arrays
            xt::view(batch_data, i - start_index) = data;
            try {
                if (this->ptr_dataset->get_label_shape().size() > 0) {
                    xt::view(batch_label, i - start_index) = label;
                } else {
                    batch_label(0) = (LType)0;
                }
            } catch (const std::bad_array_new_length &e) {
            }
            // } catch (const std::exception & e) {
            //   std::cerr << "Error retrieving data for index " << indices[i] << ": "
            //   << e.what() << std::endl;
            //   // Handle the error appropriately; you might want to return a partial
            //   batch return Batch < DType, LType > (batch_data, batch_label);
            // }
        }

        return Batch< DType, LType >(batch_data, batch_label);
    }

    Iterator begin() { return Iterator(*this, 0); }

    Iterator end() {
        // int dataset_len = ptr_dataset->len();
        // int total_batches =
        //     (dataset_len > batch_size) ? dataset_len / batch_size : 1;
        // if (dataset_len < batch_size && total_batches == 1 && drop_last) {
        //   total_batches = 0;
        // }
        return Iterator(*this, total_batches);
    }

    virtual ~DataLoader() {}

    // /////////////////////////////////////////////////////////////////////////
    // // The section for supporting the iteration and for-each to DataLoader //
    // /// START: Section                                                     //
    // /////////////////////////////////////////////////////////////////////////

    // /*TODO: Add your code here to support iteration on batch*/

    // /////////////////////////////////////////////////////////////////////////
    // // The section for supporting the iteration and for-each to DataLoader //
    // /// END: Section                                                       //
    // /////////////////////////////////////////////////////////////////////////
};

#endif /* DATALOADER_H */