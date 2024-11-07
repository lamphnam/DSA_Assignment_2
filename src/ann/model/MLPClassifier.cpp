/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this template
 */

/* 
 * File:   Model.cpp
 * Author: ltsach
 * 
 * Created on September 1, 2024, 5:09 PM
 */

#include "model/MLPClassifier.h"
#include "optim/IParamGroup.h"
#include "tensor/xtensor_lib.h"
#include "sformat/fmt_lib.h"
#include <filesystem> //require C++17
namespace fs = std::filesystem;
#include <fstream>
#include <sstream>
#include "ann/functions.h"
#include "layer/FCLayer.h"
#include "layer/ReLU.h"
#include "layer/Sigmoid.h"
#include "layer/Tanh.h"
#include "layer/Softmax.h"
#include "metrics/ClassMetrics.h"




//Constructors and Destructors
MLPClassifier::MLPClassifier(string cfg_filename, string sModelName):
    IModel(cfg_filename, sModelName){
}
MLPClassifier::MLPClassifier(
    string cfg_filename, string sModelName,
    ILayer** seq, int size): 
    IModel(cfg_filename, sModelName){
    //layer to m_layers:
    for(int idx=0; idx < size; idx++) m_layers.add(seq[idx]);
}

MLPClassifier::MLPClassifier(const MLPClassifier& orig):
    IModel(orig.m_cfg_filename, orig.m_sModelName){
    //copy list (in the assignment operator of DLinkedList)
    m_layers = orig.m_layers; 
}

MLPClassifier::~MLPClassifier() {
    for(auto ptr_layer: m_layers) delete ptr_layer;
}

// First predict method - for direct tensor input
double_tensor MLPClassifier::predict(double_tensor X, bool make_decision) {
    // Chuyển sang chế độ suy luận (inference mode)
    bool old_mode = this->m_trainable;
    this->set_working_mode(false);
    
    // Tính toán forward qua từng lớp
    double_tensor Y = X;
    for (auto& layer : m_layers) {
        Y = layer->forward(Y);
    }

    // Khôi phục chế độ ban đầu
    this->set_working_mode(old_mode);

    // Trả về nhãn hoặc xác suất theo make_decision
    if (make_decision) return xt::argmax(Y, -1);
    return Y;
}


// Second predict method - for DataLoader input
double_tensor MLPClassifier::predict(DataLoader<double, double>* pLoader, bool make_decision) {
    bool old_mode = this->m_trainable;
    this->set_working_mode(false);
    
    double_tensor results;
    bool first_batch = true;
    
    cout << "Prediction: Started" << endl;
    string info = fmt::format("{:<6s}/{:<12s}|{:<50s}\n",
                    "Batch", "Total Batch", "Num. of samples processed");
    cout << info;
    
    int total_batch = pLoader->get_total_batch(); 
    int batch_idx = 1;
    unsigned long long nsamples = 0;
    
    for(auto batch: *pLoader) {
        // Lấy data từ batch sử dụng phương thức getData()
        auto X = batch.getData();
        
        // Forward pass qua network
        double_tensor batch_output = this->forward(X);
        
        // Cập nhật results
        if(first_batch) {
            results = batch_output;
            first_batch = false;
        } else {
            results += batch_output;  // Cộng dồn kết quả
        }
        
        // Cập nhật thông tin tiến trình
        nsamples += X.shape(0);
        cout << fmt::format("{:<6d}/{:<12d}|{:<50llu}\r", 
                batch_idx++, total_batch, nsamples);
        cout.flush();
    }
    
    cout << "Prediction: End" << endl;
    
    //restore the old mode
    this->set_working_mode(old_mode);
    
    if(make_decision) return results;
    else return xt::argmax(results, -1);
}

double_tensor MLPClassifier::evaluate(DataLoader<double, double>* pLoader){
    bool old_mode = this->m_trainable;
    this->set_working_mode(false);  // Set to evaluation mode
    
    ClassMetrics meter(this->get_num_classes());
    meter.reset_metrics();
    
    // Evaluate each batch
    for (auto it = pLoader->begin(); it != pLoader->end(); ++it) {
        // Step 1: Get batch data
        Batch<double, double> batch = *it;  // Lấy batch object từ iterator
        double_tensor X = batch.getData();
        double_tensor t = batch.getLabel();
    
        // Step 2: Forward pass to get predictions
        double_tensor Y = this->forward(X);
    
        // Step 3: Calculate true and predicted labels
        ulong_tensor y_true = xt::argmax(t, 1);  // Get true class indices
        ulong_tensor y_pred = xt::argmax(Y, 1);  // Get predicted class indices
    
        // Step 4: Update metrics
        meter.accumulate(y_true, y_pred);
    }  
    
    // Get final metrics
    double_tensor metrics = meter.get_metrics();
    
    // Restore original training mode
    this->set_working_mode(old_mode);
    return metrics;
}
//for the inference mode:end



//for the training mode:begin
void MLPClassifier::compile(
                IOptimizer* pOptimizer,
                ILossLayer* pLossLayer, 
                IMetrics* pMetricLayer){
    this->m_pOptimizer = pOptimizer;
    this->m_pLossLayer = pLossLayer;
    this->m_pMetricLayer = pMetricLayer;
    
    for(auto pLayer: m_layers){
        if(pLayer->has_learnable_param()){
            string name = pLayer->getname();
            IParamGroup* pGroup = pOptimizer->create_group(name);
            pLayer->register_params(pGroup);
        }
    }
}
    
void MLPClassifier::set_working_mode(bool trainable){
    m_trainable = trainable;
    for(auto pLayer: m_layers){
        pLayer->set_working_mode(trainable);
    }
}

//protected: for the training mode: begin
double_tensor MLPClassifier::forward(double_tensor X) {
    double_tensor Y = X;
    
    for (auto& layer : m_layers) {
        Y = layer->forward(Y);
    }
    
    return Y;
}

void MLPClassifier::backward() {
    double_tensor DY = m_pLossLayer->backward();
    
    // Thêm gradient clipping
    double clip_value = 5.0; // Có thể điều chỉnh giá trị này
    DY = xt::clip(DY, -clip_value, clip_value);
    
    for (auto layer = m_layers.bbegin(); layer != m_layers.bend(); ++layer) {
        DY = (*layer)->backward(DY);
        // Clip gradient cho mỗi layer
        DY = xt::clip(DY, -clip_value, clip_value);
    }
}

//protected: for the training mode: end


/*
 * save(string base_path):
 *  + base_path: 
 
 */
bool MLPClassifier::save(string model_path){
    try{
        //prepare folder and file names
        model_path = trim(model_path);
        if(fs::exists(model_path)){
            //USE the specified path
            fs::remove_all(model_path); //remove all files related
        }
        else{
            //USE the DEFAULT path
            model_path = m_pConfig->get_new_checkpoint(this->m_sModelName);

        }
        cout << model_path << ": creation" << endl;
        fs::create_directories(model_path);
        /*code goc: string arch_file =  fs::path(model_path)/
                            fs::path(m_pConfig->get("arch_file", "arch.txt"));*/
        string arch_file =  (fs::path(model_path)/
                            fs::path(m_pConfig->get("arch_file", "arch.txt"))).string();

        //open stream
        ofstream datastream(arch_file);
        if(!datastream.is_open()){
            cerr << arch_file << ": couldn't open for writing" << endl;
            throw "Model architecture file '" + arch_file + "': can not open for writing.";
        }
        //write header
        //write data
        datastream << "model name: " << this->m_sModelName << endl;
        for(auto pLayer: m_layers){
            string desc = pLayer->get_desc();
            datastream << desc << endl;
            pLayer->save(model_path);
        }

        //close stream
        datastream.close();
        return true;
    }
    catch(exception& e){
        string message = fmt::format("MLPClassifier::save: failed; model_path={:s}",
                model_path);
        cerr << message << endl;
        cerr << e.what() << endl;
        return false;
    }
}

bool MLPClassifier::load(string model_path,  bool use_name_in_file){
    try{
        //verify the existing of model_path
        if(!fs::exists(model_path)){
            string message = fmt::format("{:s}: not exist.", model_path); 
            cerr << message << endl;
            return false;
        }

        //open a stream for the architecture file
        string arch_file = model_path + "/" + "arch.txt";
        ifstream datastream(arch_file);
        if(!datastream.is_open()){
            string message = fmt::format("{:s}: can not open for reading.", arch_file); 
            cerr << message << endl;
            return false;
        }
        //read header: to be here

        //read and parse lines
        string line;
        while(getline(datastream, line)){
            //skip empty and comment line (started with #)
            line = trim(line);
            if(line.size() == 0) continue;
            if(line[0] == '#') continue;

            //parse line
            char delimiter=':';
            istringstream linestream(line);
            string first, second;
            getline(linestream, first, delimiter); //first: maybe an empty string
            getline(linestream, second, delimiter); //second: maybe an empty string

            delimiter=',';
            istringstream partstream(first);
            string layer_type, layer_name;
            getline(partstream, layer_type, delimiter); //type: maybe an empty string
            getline(partstream, layer_name, delimiter); //name: maybe an empty string
            layer_type = trim(layer_type);
            layer_name = trim(layer_name);

            //create layers according to the layer type
            string new_name;
            if(use_name_in_file) new_name = layer_name;
            else new_name = "";

            if(layer_type.compare("FC") == 0){
                string w_file = model_path + "/" + layer_name + "_W.npy";
                string b_file = model_path + "/" + layer_name + "_b.npy";
                //note:: b_file: may not be used in FCLayer
                 m_layers.add(new FCLayer(trim(second), w_file, b_file, new_name));
            }
            if(layer_type.compare("ReLU") == 0){
                m_layers.add(new ReLU(new_name) );
            }
            if(layer_type.compare("Sigmoid") == 0){
                m_layers.add(new Sigmoid(new_name) );
            }
            if(layer_type.compare("Tanh") == 0){
                m_layers.add(new Tanh(new_name) );
            }
            if(layer_type.compare("Softmax") == 0){
                int nAxis;
                try{
                    nAxis = stoi(trim(second));
                }
                catch(std::invalid_argument& e){
                    string message_1 = fmt::format("Can not read axis of Softmax from: {:s}", trim(second));
                    string message_2 = "Use 'axis=-1' instead.";
                    cerr << message_1 << endl;
                    cout << message_2 << endl;
                    nAxis = -1; 
                }
                m_layers.add(new Softmax(nAxis, new_name) );
            }
        }
        
        //close stream
        datastream.close();
        return true;
    }
    catch(exception& e){
        cerr << "In MLPClassifier::load(.,.):" << endl;
        cout << e.what() << endl;
        return false;
    }
    return true;
}

