#include "../../CNN/model.h"
#include <Dataset/MNIST.h>
#include<Dataset.h>

int main()
{
    vector<layer_t* > layers;

    tdsize batch_size = {-1,28,28,1};

    conv_layer_t * layer1c = new conv_layer_t(1,5,64,batch_size, false, false);
    prelu_layer_t * layer1p = new prelu_layer_t( layer1c->out_size, false, false);
    batch_norm_layer_t * layer1b = new batch_norm_layer_t(layer1p->out_size);

    conv_layer_bin_t * layer2cb = new conv_layer_bin_t(1,5,128,layer1b->out_size, false, false);
    prelu_layer_t * layer2p = new prelu_layer_t( layer2cb->out_size, false, false);
    batch_norm_layer_t * layer2b = new batch_norm_layer_t(layer2p->out_size);

    conv_layer_bin_t * layer3cb = new conv_layer_bin_t(1,5,192,layer2b->out_size, false, false);
    prelu_layer_t * layer3p = new prelu_layer_t( layer3cb->out_size, false, false);
    batch_norm_layer_t * layer3b = new batch_norm_layer_t(layer3p->out_size);
    
    conv_layer_bin_t * layer4cb = new conv_layer_bin_t(1,5,256,layer3b->out_size, false, false);
    prelu_layer_t * layer4p = new prelu_layer_t( layer4cb->out_size, false, false);
    batch_norm_layer_t * layer4b = new batch_norm_layer_t(layer4p->out_size);

    conv_layer_bin_t * layer5cb = new conv_layer_bin_t(1,5,128,layer4b->out_size, false, false);
    prelu_layer_t * layer5p = new prelu_layer_t( layer5cb->out_size, false, false);
    batch_norm_layer_t * layer5b = new batch_norm_layer_t(layer5p->out_size);

    fc_layer_t * layer6 = new fc_layer_t(layer5b->out_size, {-1, 10, 1, 1});
    softmax_layer_t * layer7 = new softmax_layer_t(layer6->out_size,false);

    layers.push_back((layer_t *) layer1c);
    layers.push_back((layer_t *) layer1p);
    layers.push_back((layer_t *) layer1b);
    layers.push_back((layer_t *) layer2cb);
    layers.push_back((layer_t *) layer2p);
    layers.push_back((layer_t *) layer2b);
    layers.push_back((layer_t *) layer3cb);
    layers.push_back((layer_t *) layer3p);
    layers.push_back((layer_t *) layer3b);
    layers.push_back((layer_t *) layer4cb);
    layers.push_back((layer_t *) layer4p);
    layers.push_back((layer_t *) layer4b);
    layers.push_back((layer_t *) layer5cb);
    layers.push_back((layer_t *) layer5p);
    layers.push_back((layer_t *) layer5b);
    layers.push_back((layer_t *) layer6);
    layers.push_back((layer_t *) layer7);






    
    // tensor_t<float>::ccount = 0;
    // tensor_t<float>::dcount = 0;
    Model model(layers);
    model.summary();

    string PATH="trained_models/big_binary_mnist_64x";

    #ifdef using_cmake
    PATH="../Example\\ MNIST/full_test/trained_models/big_binary_mnist_64x";
    #endif

    Dataset data = load_mnist(90,10,0);
    // model.load("PATH");

    for(int ep = 0; ep<10; ep++){
        
        // float nlr = model.Step_decay(ep);

        model.train(data.train.images, data.train.labels, 16, 1, 0.0001);

        tensor_t<float> output = model.predict(data.test.images, 1);
        int correct = 0;

        for(int i=0; i<output.size.m; i++){
            int idx,aidx;
            float maxm = 0.0f;

            for(int j=0; j<10; j++){
                if(output(i,j,0,0)>maxm){
                    maxm = output(i,j,0,0);
                    idx = j;
                }
                if(int(data.test.labels(i,j,0,0))==1){
                    aidx = j;
                }
            }
            if(idx == aidx) correct++;
            // cout<<"predicted: "<<idx<<" actual: "<<aidx<<endl;
        }

        cout<<"correct number is "<<correct<< " / " << output.size.m << endl;
        cout<<"accuracy "<<correct/output.size.m<<endl;

    }
   

    // model.train(data.train.images, data.train.labels, 56, 2, 0.0002);
    
    model.save(PATH);

    // model.save(PATH);


    tensor_t<float> output = model.predict(data.test.images, 1);
    int correct = 0;

    for(int i=0; i<output.size.m; i++){
        int idx,aidx;
        float maxm = 0.0f;

        for(int j=0; j<10; j++){
            if(output(i,j,0,0)>maxm){
                maxm = output(i,j,0,0);
                idx = j;
            }
            if(int(data.test.labels(i,j,0,0))==1){
                aidx = j;
            }
        }
        if(idx == aidx) correct++;
        // cout<<"predicted: "<<idx<<" actual: "<<aidx<<endl;
    }

    cout<<"correct number is "<<correct<< " / " << output.size.m << endl;
    cout<<"accuracy "<<correct/output.size.m<<endl;

    // model.train(packet.data, packet.out, 16, 2, 0.001);
    // model.save_weights("weights_after_1_epoch_lr_1e-5_b32");
    // model.train(packet.data, packet.out, 16, 2, 0.00001);
    // model.save_weights("weights_after_3_epochs_lr_1e-5_b32");
    // mod el.train(packet.data, packet.out, 16, 100, 0.00001);
    // model.save_weights("weights_after_13_epochs_lr_1e-4_b32");
    // model.train(packet.data, packet.out, 16, 10, 0.00001);
    // model.save_weights("weights_after_23_epochs_lr_1e-5_b32");
    // model.train(packet.data, packet.out, 16, 100, 0.00001);
    // model.save_weights("weights_after_unlimited_fully_on_lr_1e-5_b32");
    return 0;

}

