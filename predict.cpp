#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>

// ANSI color codes
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

class NeuralNetwork {
public:
    double weights[4][3];
    double bias[3];

    NeuralNetwork() {}

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    std::vector<double> predict(const std::vector<double>& inputs) {
        std::vector<double> outputs(3, 0.0);
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 4; ++i) {
                outputs[j] += inputs[i] * weights[i][j];
            }
            outputs[j] += bias[j];
            outputs[j] = sigmoid(outputs[j]);
        }
        return outputs;
    }

    void load_model(const std::string& filename) {
        std::ifstream file(filename);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                file >> weights[i][j];
            }
        }
        for (int j = 0; j < 3; ++j) {
            file >> bias[j];
        }
        file.close();
    }
};

void print_centered(const std::string& text, char fill = '=', int width = 60) {
    int padding = (width - text.length()) / 2;
    std::cout << std::string(padding, fill) << text << std::string(padding, fill) << std::endl;
}

double get_double_input(const std::string& prompt) {
    double value;
    std::string input;
    while (true) {
        std::cout << prompt;
        std::getline(std::cin, input);
        std::istringstream iss(input);
        if (iss >> value && iss.eof()) {
            return value;
        } else {
            std::cout << RED << "Invalid input. Please enter a valid number." << RESET << std::endl;
        }
    }
}

void print_progress_bar(double percentage, int width = 30) {
    int pos = width * percentage;
    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(percentage * 100.0) << "%\r";
    std::cout.flush();
}

// Simple delay function
void delay() {
    for(int i = 0; i < 10000000; ++i) {
        // Do nothing, just waste some cycles
    }
}

int main() {
    std::cout << "\033[2J\033[1;1H"; // Clear screen
    print_centered(BOLD MAGENTA "Iris Species Predictor" RESET, '=', 60);
    std::cout << std::endl;

    NeuralNetwork nn;
    nn.load_model("model.txt");

    std::vector<std::string> species_names = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

    while (true) {
        std::cout << CYAN << "\nEnter four input features (or type 'exit' to quit):" << RESET << std::endl;
        std::cout << YELLOW << "1. Sepal length (cm): " << RESET;
        std::string input;
        std::getline(std::cin, input);
        
        if (input == "exit") {
            break;
        }

        double x1 = std::stod(input);
        double x2 = get_double_input(YELLOW "2. Sepal width (cm):  " RESET);
        double x3 = get_double_input(YELLOW "3. Petal length (cm): " RESET);
        double x4 = get_double_input(YELLOW "4. Petal width (cm):  " RESET);

        std::vector<double> inputs = {x1, x2, x3, x4};

        std::cout << "\n" << BLUE << "Predicting..." << RESET << std::endl;
        for (int i = 0; i <= 100; ++i) {
            print_progress_bar(i / 100.0);
            delay(); // Use our simple delay function instead of sleep
        }
        std::cout << std::endl;

        std::vector<double> prediction = nn.predict(inputs);
        int predicted_class = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
        
        std::cout << "\n" << GREEN << "Prediction Results:" << RESET << std::endl;
        std::cout << GREEN << "-------------------" << RESET << std::endl;
        std::cout << BLUE << "Predicted Species: " << RESET << BOLD << species_names[predicted_class] << RESET << std::endl;
        std::cout << BLUE << "\nProbabilities:" << RESET << std::endl;
        for (size_t i = 0; i < species_names.size(); ++i) {
            std::cout << std::setw(16) << std::left << species_names[i] << ": ";
            print_progress_bar(prediction[i], 20);
            std::cout << " " << std::fixed << std::setprecision(2) << prediction[i] * 100 << "%" << std::endl;
        }
        std::cout << std::endl;
    }

    print_centered(CYAN "Thank you for using the Iris Species Predictor. Goodbye!" RESET, '-', 70);

    return 0;
}