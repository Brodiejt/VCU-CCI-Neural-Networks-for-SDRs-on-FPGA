#ifndef RNN_HPP
#define RNN_HPP

#include <Eigen>
#include <vector>
#include <math.h>
//types

//typedef Eigen::MatrixXd MatrixXd;

class RNN
{
private:
    void forward();
    void backward();
    void loss();
    void updateWeights();
    float sigmoid(float);
    float softmax(float);
    float tanh(float);
public:
    RNN();
    void train();
};

class gru_cell
{
private:
    // reset weights and bias
    float w_r, u_r, b_r;
    // update weights and bias
    float w_z, u_z, b_z;
    // hidden_hat weights and bias
    float w_hh, u_hh, b_hh;
    // reset, update, hidden_hat, hidden
    float r, z, hh, h;

public:
    void calcHiddenState(float in, float prev_hs, float *hs);
    float sigmoid(float x) {return x / (1 + abs(x));}
    float getHiddenState(){return h;}
};
#endif