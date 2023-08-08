#include "rnn.hpp"
#include <iostream>
#include <Eigen>
#include <vector>
#include <math.h>

using namespace Eigen;

// void getH(float, float, float*);
void sigmoid(Matrix<float, Dynamic, Dynamic> &m);
void sigmoidInverse(Matrix<float, Dynamic, Dynamic> &m);
void grutanh(Matrix<float, Dynamic, Dynamic> &);

typedef struct gru_weights
{
    // reset weights and bias
    Matrix<float, Dynamic, Dynamic> w_r, u_r, b_r;
    // update weights and bias
    Matrix<float, Dynamic, Dynamic> w_z, u_z, b_z;
    // hidden_hat weights and bias
    Matrix<float, Dynamic, Dynamic> w_hh, u_hh, b_hh;

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
    // sigmoid(m);
    std::cout << m << '\n';
    return 0;
}

void getH(Matrix2f &in, Matrix2f &prev_hs, gru_weights &g, Matrix2f &hs)
{
    Matrix2f r = (g.w_r * in) + (g.u_r * prev_hs);
    sigmoid(r);
    Matrix2f z = (g.w_z * in) + (g.u_z * prev_hs);
    sigmoid(z);
    sigmoidInverse(z);
    Matrix2f hhat = (g.w_hh * in) + r.cwiseProduct(g.u_hh * prev_hs);
    grutanh(hhat);
    hs = z.cwiseProduct(prev_hs) + z.cwiseProduct(hhat);
}

void sigmoid(Matrix2f &m)
{
    for (auto &a : m.reshaped())
        a = 1 / (1 + exp(-a));
}

void grutanh(Matrix2f &m)
{
    for (auto &a : m.reshaped())
        a = tanh(a);
}

void sigmoidInverse(Matrix2f &m)
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