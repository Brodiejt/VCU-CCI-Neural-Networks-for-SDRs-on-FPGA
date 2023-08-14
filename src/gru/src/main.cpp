#include "rnn.hpp"
#include <iostream>
#include <Eigen>
#include <vector>
#include <math.h>

using namespace Eigen;

// void getH(float, float, float*);
void sigmoid(MatrixXf &m);
void sigmoidInverse(MatrixXf &m);
void grutanh(MatrixXf &);
void prepareBatch(MatrixXf &m);
MatrixXf softmax(MatrixXf &m);

typedef struct gru_weights
{
    // reset weights and bias
    MatrixXf w_r, u_r, b_r;
    // Matrix<float, 1, Dynamic> b_r;
    // update weights and bias
    MatrixXf w_z, u_z, b_z;
    // Matrix<float, 1, Dynamic> b_z;
    // hidden_hat weights and bias
    MatrixXf w_hh, u_hh, b_hh;
    // Matrix<float, 1, Dynamic> b_hh;

    // hidden states
    MatrixXf h_prev, hs;

    // output weights
    MatrixXf w_y, b_y;

} gru_weights;

int main()
{
    MatrixXf m = MatrixXf::Random(1, 7);

    std::cout << m << '\n';
    m = m * 2.0;
    sigmoid(m);
    std::cout << m << '\n';

    const int batchSize = 2;
    const int seq_len = 3;

    m.resize(batchSize, seq_len);
    m.transposeInPlace();

    std::cout << m;

    return 0;
}

void getH(MatrixXf &in, MatrixXf &prev_hs, gru_weights &g)
{
    MatrixXf r = (g.w_r * in) + (g.u_r * prev_hs);
    sigmoid(r);
    MatrixXf z = (g.w_z * in) + (g.u_z * prev_hs);
    sigmoid(z);
    sigmoidInverse(z);
    MatrixXf hhat = (g.w_hh * in) + r.cwiseProduct(g.u_hh * prev_hs);
    grutanh(hhat);
    g.hs = z.cwiseProduct(prev_hs) + z.cwiseProduct(hhat);
}

void sigmoid(MatrixXf &m)
{
    for (auto &a : m.reshaped())
        a = 1 / (1 + exp(-a));
}

void grutanh(MatrixXf &m)
{
    for (auto &a : m.reshaped())
        a = tanh(a);
}

void sigmoidInverse(MatrixXf &m)
{
    for (auto &a : m.reshaped())
        a = 1 - a;
}

MatrixXf softmax(MatrixXf &m1)
{
    MatrixXf m = m1;
    float ymax = 0;
    for (auto &a : m.reshaped())
    {
        ymax = (a > ymax) ? a : ymax;
    }

    for (auto &a : m.reshaped())
    {
        a = exp(a - ymax);
    }
    m /= m.sum();
    return m;
}
void initialize_gru_weights(gru_weights &g, int h_size, int numClass, int batchSize)
{
    // Weight matrices for inputs
    g.w_z = MatrixXf::Random(numClass, h_size);
    g.w_r = MatrixXf::Random(numClass, h_size);
    g.w_hh = MatrixXf::Random(numClass, h_size);

    // hidden states
    g.hs = MatrixXf::Zero(batchSize, h_size);
    g.h_prev = g.hs;

    // Weight matrices for hidden layer
    g.u_z = MatrixXf::Random(h_size, h_size);
    g.u_r = MatrixXf::Random(h_size, h_size);
    g.u_hh = MatrixXf::Random(h_size, h_size);

    // bias vectors
    g.b_z = MatrixXf::Random(1, h_size);
    g.b_r = MatrixXf::Random(1, h_size);
    g.b_hh = MatrixXf::Random(1, h_size);

    // output weights
    g.w_y = MatrixXf::Random(h_size, numClass);
    g.b_y = MatrixXf::Zero(numClass);
}

// Forward pass
void gru(MatrixXf &in, MatrixXf &h, gru_weights weights)
{
    MatrixXf y_t;
    for (int i = 0; i < in.rows(); i++)
    {

        weights.h_prev = h;
        getH(in, weights.h_prev, weights);

        // Calc linear layer
        MatrixXf y_linear = (weights.hs * weights.w_y) + weights.b_y;
        // Calc y_t with softmax activation function

        y_t = softmax(y_linear);
    }
}