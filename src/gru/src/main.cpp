#include "rnn.hpp"
#include <iostream>
#include <Eigen>
#include <Eigen/Dense>
#include <vector>
#include <math.h>

using namespace Eigen;

// void getH(float, float, float*);
void sigmoid(Eigen::Matrix2f &);
void sigmoidInverse(Eigen::Matrix2f &m);
void grutanh(Eigen::Matrix2f &);
typedef struct gru_weights
{
    // reset weights and bias
    Eigen::Matrix2f w_r, u_r, b_r;
    // update weights and bias
    Eigen::Matrix2f w_z, u_z, b_z;
    // hidden_hat weights and bias
    Eigen::Matrix2f w_hh, u_hh, b_hh;
} gru_weights;

int main()
{
    Eigen::Matrix2f m(2, 2);
    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(1, 0) = 3.0;
    m(1, 1) = 4.0;
    std::cout << m << '\n';
    m = m * 2.0;
    sigmoid(m);
    std::cout << m << '\n';
    return 0;
}

void getH(Eigen::Matrix2f &in, Eigen::Matrix2f &prev_hs, gru_weights &g, Eigen::Matrix2f &hs)
{
    Eigen::Matrix2f r = (g.w_r * in) + (g.u_r * prev_hs);
    sigmoid(r);
    Eigen::Matrix2f z = (g.w_z * in) + (g.u_z * prev_hs);
    sigmoid(z);
    sigmoidInverse(z);
    Eigen::Matrix2f hhat = (g.w_hh * in) + r.cwiseProduct(g.u_hh * prev_hs);
    grutanh(hhat);
    hs = z.cwiseProduct(prev_hs) + z.cwiseProduct(hhat);
}

void sigmoid(Eigen::Matrix2f &m)
{
    for (auto &a : m.reshaped())
        a = 1 / (1 + exp(-a));
}

void grutanh(Eigen::Matrix2f &m)
{
    for (auto &a : m.reshaped())
        a = tanh(a);
}

void sigmoidInverse(Eigen::Matrix2f &m)
{
    for (auto &a : m.reshaped())
        a = 1 - a;
}
void initialize_gru_weights(gru_weights &g, int h_size, int numClass, int batchSize)
{
    // Weight matrices for inputs
    g.w_z = Matrix2f::Random(numClass, h_size);
    g.w_r = Matrix2f::Random(numClass, h_size);
    g.w_hh = Matrix2f::Random(numClass, h_size);

    // hidden state
    Eigen::Matrix2f h_state = Matrix2f::Zero(batchSize, h_size);

    // Weight matrices for hidden layer
    g.u_z = Matrix2f::Random(h_size, h_size);
    g.u_r = Matrix2f::Random(h_size, h_size);
    g.u_hh = Matrix2f::Random(h_size, h_size);

    // bias vectors
    g.b_z = Matrix2f::Random(h_size);
    g.b_r = Matrix2f::Random(h_size);
    g.b_hh = Matrix2f::Random(h_size);
}

// Forward pass
void gru(Eigen::Matrix2f &in, int idden_size)
{
    Eigen::Matrix2f outputs;
    for (auto &sequence : in.reshaped())
    {
        // getH(sequence, , )
    }
}