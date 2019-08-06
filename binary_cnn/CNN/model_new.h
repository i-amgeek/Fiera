#include <cassert>
#include <cstdint>
#include <string.h>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <chrono>  // for high_resolution_clock
#include <sys/stat.h>
#include "byteswap.h"
#include "cnn_new.h"

using namespace std;
#pragma pack(push, 1)

class Model{

    vector<layer_t* > layers;
    int epochs=0;
    int batch_size;
    int num_of_batches;
    float loss;
    float learning_rate;

    public:
        Model(vector<layer_t *> layers){
            this->layers = layers;
        }
        Model(){}

        float Step_decay(float epoch){
            float drop = 0.9;
            float epoch_drop = 1;

            float n_learning_rate = learning_rate*pow(drop, floor((1+epoch)/epoch_drop)); 
            return n_learning_rate;
        }
        void train( xarray<float> input, xarray<float> output, int batch_size, int epochs=1, float lr = 0.02, string optimizer="Momentum", string lr_schedule = "Step_decay", bool debug=false ){
        //TODO: Check layers are not empty
        
            
            this->epochs += epochs;
            this->batch_size = batch_size;
            this->num_of_batches = input.shape()[0] / batch_size ;
            this->learning_rate = lr;


            cout<<"Total images: " << input.shape()[0]<<endl;
            cout<<"batch_size: " << batch_size<<endl;

            for ( int epoch = 0; epoch < epochs; ++epoch){
                for(int batch_num = 0; batch_num<num_of_batches; batch_num++)
                {
                    auto start = std::chrono::high_resolution_clock::now();

                    int start_index = batch_num * batch_size;
                    int end_index = start_index + batch_size;
                    
                    xarray<float> input_batch = xt::view(input, xt::range(start_index, end_index));
                    xarray<float> labels_batch = xt::view(output, xt::range(start_index, end_index));
                    xarray<float> out;

                    // Forward propogate
                    for ( int i = 0; i < layers.size(); i++ )
                    {
                        if ( i == 0 )
                            out = activate( layers[i], input_batch, true);
                        else
                            out = activate( layers[i], out, true);
                    }

                    // Calculate Loss
                    this->loss = cross_entropy(out, labels_batch, debug);
                    
                    cout <<"loss for epoch: "<< epoch << "/" << epochs << " and batch: " << batch_num << "/" << num_of_batches << " is " << loss << endl;

                    // Backpropogation
                    xarray<float> grads_in;

                    for ( int i = layers.size() - 1; i >= 0; i-- )
                    {

                        if ( i == layers.size() - 1 )
                            grads_in = calc_grads( layers[i], labels_batch);
                        else
                            grads_in = calc_grads( layers[i], grads_in );
                        
                     }
                    
                    float n_lr;

                    if(lr_schedule == "Step_decay")
                      n_lr = Step_decay(epoch);
                    else 
                        n_lr = learning_rate;
                        
                    // Update weights
                    for ( int i = 0; i < layers.size(); i++ )
                        fix_weights( layers[i], n_lr);
                    
                    if (!(batch_num%10)){ 
                        auto finish = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> elapsed = finish - start;
                        cout << "Estimated time: " << elapsed.count() / 60.0 * ((num_of_batches - batch_num + 1) + num_of_batches * (epochs - epoch + 1)) <<" minutes" << endl;
                    }
                }
            }
        }

        xarray<float> predict(xarray<float> input_batch){

            xarray<float> out;
            auto start = std::chrono::high_resolution_clock::now();

            for ( int i = 0; i < layers.size(); i++ )
            {
                if ( i == 0 )
                    out = activate( layers[i], input_batch, false);
                else
                    out = activate( layers[i], out, false);
            }

            

            return out;
        }
        
        xarray<float> predict (string input_image){
            // Takes path of input image (only .ppm currently) as input.
            struct stat buffer;   
            assert(stat (input_image.c_str(), &buffer) == 0);   // Checks if file exist

            ifstream file( input_image, ios::binary | ios::ate ); 
            streamsize size = file.tellg();
            file.seekg( 0, ios::beg );
            assert( size != -1); 
            uint8_t* data = new uint8_t[size];
            file.read( (char*)data, size );
            if ( data )
            {
                uint8_t * usable = data;
                while ( *(uint32_t*)usable != 0x0A353532 )
                    usable++;

    #pragma pack(push, 1)
                struct RGB
                {
                    uint8_t r, g, b;
                };
    #pragma pack(pop)

                RGB * rgb = (RGB*)usable;

                auto image = xt::xarray<float>::from_shape({1, 28, 28, 1});
                xarray<float> output;
                for ( int i = 0; i < 28; i++ )
                {
                    for ( int j = 0; j < 28; j++ )
                    {
                        RGB rgb_ij = rgb[i * 28 + j];
                        image( 0, i, j, 0 ) = ((((float)rgb_ij.r
                                    + rgb_ij.g
                                    + rgb_ij.b)
                                    / (3.0f*255.f)));
                    }
				}

                for ( int i = 0; i < layers.size(); i++ )
                {
                    if ( i == 0 )
                        output = activate( layers[i], image, false);
                    else
                        output = activate( layers[i], output, false);
                return output;

                }
			}
        }


        void save_model( string fileName ){
            ofstream file(fileName);
            json model;

            // If any configuration needed later, save here.
            model["config"] = {
                {}
            };
            for ( int i = 0; i < layers.size(); i++ ) 
                save_layer( layers[i], model );
            file << std::setw(4) << model << std::endl;
            cout << "\n Model saved in file " << fileName << endl;
            file.close();
        }

        void load_model( string fileName ){
            if (!layers.empty()) {
                cout << "Deleting previous stored layers \n" << endl;
                layers.clear();
            }

            struct stat buffer;
            if (!stat (fileName.c_str(), &buffer) == 0) {
                cout << "File not found: " << fileName << endl;
                exit(0);
            }
            ifstream file(fileName);
            json model;
            file >> model;
            json layersJ = model["layers"];

            for (auto& el : layersJ.items()){
                json layerJ = el.value();
                json inJ = layerJ["in_size"];
                tdsize in_size;
                in_size.from_json(inJ);
                
                cout<<layerJ["layer_type"]<<endl;

                if (layerJ["layer_type"] == "fc"){
                    json outJ = layerJ["out_size"];
                    tdsize out_size;
                    out_size.from_json(outJ);
                    fc_layer_t * layer = new fc_layer_t(in_size, out_size );
                    layers.push_back((layer_t *) layer);
                    continue;
                } 

                else if (layerJ["layer_type"] == "conv"){
                    conv_layer_t * layer = new conv_layer_t(layerJ["stride"], layerJ["extend_filter"], layerJ["number_filters"], in_size);
                    layers.push_back((layer_t *) layer);
                    continue;
                }

                else if (layerJ["layer_type"] == "prelu"){
                    prelu_layer_t * layer = new prelu_layer_t(in_size);
                    layer->prelu_zero = layerJ["prelu_zero"];
                    layers.push_back((layer_t *) layer);
                    continue;
                }

                else if (layerJ["layer_type"] == "softmax"){
                    softmax_layer_t * layer = new softmax_layer_t(in_size);
                    layers.push_back((layer_t *) layer);
                    continue;
                } 

                else if (layerJ["layer_type"] == "batch_norm2D"){
                    batch_norm_layer_t * layer = new batch_norm_layer_t(in_size);
                    layers.push_back((layer_t *) layer);
                    continue;
                }

                else if (layerJ["layer_type"] == "conv_bin"){
                    conv_layer_bin_t * layer = new conv_layer_bin_t(layerJ["stride"], layerJ["extend_filter"], layerJ["number_filters"], in_size);
                    layers.push_back((layer_t *) layer);
                    continue;
                }

                else if (layerJ["layer_type"] == "flatten"){
                    flatten_t * layer = new flatten_t(in_size);
                    layers.push_back((layer_t *) layer);
                    continue;
                }

                // else if (layerJ["layer_type"] == "fc_bin"){
                //     json outJ = layerJ["out_size"];
                //     tdsize out_size;
                //     out_size.from_json(outJ);
                //     fc_layer_bin_t * layer = new fc_layer_bin_t(in_size, out_size);
                //     layers.push_back((layer_t *) layer);
                //     continue;
                // }

                else if (layerJ["layer_type"] == "scale"){
                    scale_layer_t * layer = new scale_layer_t(in_size);
                    layers.push_back((layer_t *) layer);
                    continue;
                }
            }
            cout << "\n Model loaded successfully from " << fileName << endl;
        }

        void save_weights( string folderName ){
            assert(layers.size() > 1);

            mkdir(folderName.c_str(), 0777);
            
            time_t now = time(0);
            string date = ctime(&now);
            json j = {
                {"Date", date},
                {"Epochs", this->epochs},
                {"Num_of_batches", this->num_of_batches},
                {"Batch_size", this->batch_size},
                {"Training_loss", this->loss},
                {"Learning_rate", this->learning_rate}
            };
            ofstream file(folderName + "/metadata.json");
            file << j;
            file.close();

            for ( int i = 0; i < layers.size(); i++ ){
                string fileName = folderName + "/" + to_string(i) + ".weights";
                save_layer_weight( layers[i], fileName );
            }        
            cout << "\n Mode saved in folder " << folderName << endl;
        }

        void load_weights( string folderName ){
            assert(layers.size() > 1);
            ifstream file(folderName+"/metadata.json");
            json j;
            file >> j;
            this->epochs = j["Epochs"];
            this->num_of_batches = j["Num_of_batches"];
            this->batch_size = j["Batch_size"];
            this->loss = j["Training_loss"];
            file.close();
            struct stat buffer;
            for ( int i = 0; i < layers.size(); i++ ){
                string fileName = folderName + "/" + to_string(i) + ".weights";
                if (stat (fileName.c_str(), &buffer) == 0)
                    load_layer_weight( layers[i], fileName);
                else{
                    cout << "File " << fileName << " does not exist\n";
                    exit(0);
                }
            }
            cout << "\n Model loaded from folder " << folderName << endl;
        }

        void load( string folderName ){
            this->load_model( folderName + "/model.json");
            this->load_weights( folderName + "/weights");
        }

        void save( string folderName ){
            mkdir(folderName.c_str(), 0777);
            this->save_model( folderName + "/model.json");
            this->save_weights( folderName + "/weights");
        }

        void summary(){
            cout << "\n\t\t\tMODEL SUMMARY\n";
            for (auto& layer : layers){
                print_layer(layer);
            }
            cout << endl<< endl;
        }
};

#pragma pack(pop)
