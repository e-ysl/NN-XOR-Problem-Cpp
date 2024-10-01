#include <iostream>
#include <fstream> 
#include <cmath>
#include <cstdlib> 
#include <ctime>   
using namespace std;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1 - x);
}

void feedforward(int g1, int g2, double w[], double biasw[], double &h1_out, double &h2_out, double &o_out) {
    double h1_in = g1 * w[0] + g2 * w[2] + biasw[0];
    h1_out = sigmoid(h1_in);

    double h2_in = g1 * w[1] + g2 * w[3] + biasw[1];
    h2_out = sigmoid(h2_in);

    double o_in = h1_out * w[4] + h2_out * w[5] + biasw[2];
    o_out = sigmoid(o_in);
}

void backpropagation(int g1, int g2, double target, double w[], double biasw[], double &h1_out, double &h2_out, double &o_out, double learning_rate) {

    feedforward(g1, g2, w, biasw, h1_out, h2_out, o_out);

    double error = target - o_out;

    double delta_o = error * sigmoid_derivative(o_out);
    w[4] += learning_rate * h1_out * delta_o;
    w[5] += learning_rate * h2_out * delta_o;
    biasw[2] += learning_rate * delta_o; 

    double delta_h1 = delta_o * w[4] * sigmoid_derivative(h1_out);
    double delta_h2 = delta_o * w[5] * sigmoid_derivative(h2_out);
    w[0] += learning_rate * g1 * delta_h1;
    w[1] += learning_rate * g1 * delta_h2;
    w[2] += learning_rate * g2 * delta_h1;
    w[3] += learning_rate * g2 * delta_h2;
    biasw[0] += learning_rate * delta_h1; 
    biasw[1] += learning_rate * delta_h2; 
}

void test(double inputs[][2], double w[], double biasw[]) {
    for (int i = 0; i < 4; ++i) {
        double g1 = inputs[i][0];
        double g2 = inputs[i][1];
        double h1_out, h2_out, o_out;
        feedforward(g1, g2, w, biasw, h1_out, h2_out, o_out);
        cout << g1 << " XOR " << g2 << " --> " << o_out << endl;
    }
}

int main() {
    srand(time(0));

    double w[] = {(static_cast<double>(rand()) / RAND_MAX) * 2 - 1, (static_cast<double>(rand()) / RAND_MAX) * 2 - 1, 
                  (static_cast<double>(rand()) / RAND_MAX) * 2 - 1, (static_cast<double>(rand()) / RAND_MAX) * 2 - 1, 
                  (static_cast<double>(rand()) / RAND_MAX) * 2 - 1, (static_cast<double>(rand()) / RAND_MAX) * 2 - 1}; 
    double biasw[] = {(static_cast<double>(rand()) / RAND_MAX) * 2 - 1, (static_cast<double>(rand()) / RAND_MAX) * 2 - 1, 
                     (static_cast<double>(rand()) / RAND_MAX) * 2 - 1}; 

    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[4] = {0, 1, 1, 0};

    double learning_rate = 0.5; 

    ofstream loss_file("loss.txt");

    for (int epoch = 0; epoch < 10000; ++epoch) { 
        double total_loss = 0.0;
        for (int i = 0; i < 4; ++i) {
            double g1 = inputs[i][0];
            double g2 = inputs[i][1];
            double target = targets[i];

            double h1_out, h2_out, o_out;
            backpropagation(g1, g2, target, w, biasw, h1_out, h2_out, o_out, learning_rate);

            double error = target - o_out;
            total_loss += error * error;
        }
        loss_file << epoch << " " << total_loss << endl;
    }

    loss_file.close();

    test(inputs, w, biasw);

    return 0;
}
