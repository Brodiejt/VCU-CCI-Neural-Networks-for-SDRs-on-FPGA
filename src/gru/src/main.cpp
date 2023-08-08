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

} gru_weights;

int main()
{
    Matrix2f m(2, 2);
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

void getH(MatrixXf &in, MatrixXf &prev_hs, gru_weights &g, MatrixXf &hs)
{
    MatrixXf r = (g.w_r * in) + (g.u_r * prev_hs);
    sigmoid(r);
    MatrixXf z = (g.w_z * in) + (g.u_z * prev_hs);
    sigmoid(z);
    sigmoidInverse(z);
    MatrixXf hhat = (g.w_hh * in) + r.cwiseProduct(g.u_hh * prev_hs);
    grutanh(hhat);
    hs = z.cwiseProduct(prev_hs) + z.cwiseProduct(hhat);
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
void initialize_gru_weights(gru_weights &g, int h_size, int numClass, int batchSize)
{
    // Weight matrices for inputs
    g.w_z = MatrixXf::Random(numClass, h_size);
    g.w_r = MatrixXf::Random(numClass, h_size);
    g.w_hh = MatrixXf::Random(numClass, h_size);

    // hidden state
    MatrixXf h_state = MatrixXf::Zero(batchSize, h_size);

    // Weight matrices for hidden layer
    g.u_z = MatrixXf::Random(h_size, h_size);
    g.u_r = MatrixXf::Random(h_size, h_size);
    g.u_hh = MatrixXf::Random(h_size, h_size);

    // bias vectors
    g.b_z = MatrixXf::Random(1,h_size);
    g.b_r = MatrixXf::Random(1,h_size);
    g.b_hh = MatrixXf::Random(1,h_size);
}

// Forward pass
void gru(Matrix2f &in, int idden_size)
{
    Matrix2f outputs;
    for (auto &sequence : in.reshaped())
    {
        // getH(sequence, , )
    }
}