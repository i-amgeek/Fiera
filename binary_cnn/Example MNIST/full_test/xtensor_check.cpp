#include "../../CNN/model.h"
#include "../../CNN/Dataset/MNIST.h"
#include"../../CNN/Dataset.h"

int main()
{
    vector<layer_t* > layers;
          
    tensor_t<float> temp_in(2, 3, 3, 2), filters(2, 2, 2, 2), expected_output(2, 2, 2, 2);
    std::vector<std::vector<std::vector<std::vector<float> > > > vect=

        {{{{-2.9662, -1.0606, -0.3090},
          { 0.9343, -0.3821, -1.1669},
          { 0.3636, -0.3156,  1.1450}},

         {{-0.3822, -0.3553,  0.7542},
          { 0.6901, -0.1443,  1.6120},
          { 1.5671, -1.2432, -1.7178}}},


        {{{-0.5824, -0.6153,  0.4105},
          { 1.7675, -0.0832,  0.5087},
          { 1.1178,  1.1286,  0.1416}},

         {{-0.5458,  1.1542, -1.5366},
          {-0.5577, -0.4383,  1.1572},
          { 0.0889,  0.2659, -0.1907}}}};

    temp_in.from_vector(vect);
        
    for(int i=0; i<4; i++) 
        for(int img=0;img<2;img++) 
            predict(img,i,0,0) = 0;
    predict(0,3,0,0) = 1;
    predict(1 ,1, 0, 0) = 1;

    tdsize batch_size = {-1,4,4,2};

    // conv_layer_t * layer1b = new conv_layer_t(1,3,14,batch_size, false, false);
    // prelu_layer_t * layer2p = new prelu_layer_t( layer1b->out_size, false, false);
    // batch_norm_layer_t * layerbaa = new batch_norm_layer_t(layer2p->out_size);
    



    conv_layer_bin_t * layer1bb = new conv_layer_bin_t(1,2,2,batch_size, false, false);
    layer1bb->filters.from_vector(vect);


    // layers.push_back((layer_t *) layer1b);
    // layers.push_back((layer_t *) layer2p);
    // layers.push_back((layer_t *) layerbaa);
    layers.push_back((layer_t *) layer1bb);
    
    Model model(layers);
    model.summary();

    /* string PATH="trained_models/big_binary_mnist"; */

    /* #ifdef using_cmake */
    /* PATH="Example\\ MNIST/full_test/trained_models/big_binary_mnist"; */
    /* #endif */

    /* // Dataset data = load_mnist(60,10,0); */
    /* // model.load("PATH"); */

    /* model.train(temp_in, predict, 1, 20, 0.0001); */

    // model.train(data.train.images, data.train.labels, 56, 2, 0.0002);
    

    // model.save(PATH);
    // model.save(PATH);


    tensor_t<float> output = model.predict(temp_in, 1);
    // int correct = 0;

    // for(int i=0; i<output.size.m; i++){
    //     int idx,aidx;
    //     float maxm = 0.0f;

    //     for(int j=0; j<10; j++){
    //         if(output(i,j,0,0)>maxm){
    //             maxm = output(i,j,0,0);
    //             idx = j;
    //         }
    //         if(int(data.test.labels(i,j,0,0))==1){
    //             aidx = j;
    //         }
    //     }
    //     if(idx == aidx) correct++;
    //     cout<<"predicted: "<<idx<<" actual: "<<aidx<<endl;
    // }

    // cout<<"correct number is "<<correct<< " / " << output.size.m << endl;

    // cout<<"******Constructor called********* "<<tensor_t<float>::ccount<<endl;
    // cout<<"*****Destructor called*********** "<<tensor_t<float>::dcount<<endl;

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
