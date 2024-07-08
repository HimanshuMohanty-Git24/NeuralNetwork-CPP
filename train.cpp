#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <iomanip>
#include <numeric>
#include <thread>

// ANSI color codes
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

// Debug macro
#define DEBUG(x) std::cout << YELLOW << "DEBUG: " << x << RESET << std::endl

// Utility function to read the dataset from a CSV file
std::vector<std::vector<double>> read_dataset(const std::string &filename, std::vector<int> &labels, std::vector<std::string> &label_names) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::vector<std::vector<double>> dataset;
    std::string line;
    std::unordered_map<std::string, int> label_map;
    int label_counter = 0;

    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> data_point;
        std::string value;

        // Skip the ID column
        std::getline(ss, value, ',');

        // Read the 4 feature values
        for (int i = 0; i < 4; ++i) {
            if (!std::getline(ss, value, ',')) {
                throw std::runtime_error("Error reading CSV: unexpected end of line");
            }
            try {
                data_point.push_back(std::stod(value));
            } catch (const std::exception& e) {
                throw std::runtime_error("Error converting to double: " + value);
            }
        }

        // Read the species label
        if (!std::getline(ss, value)) {
            throw std::runtime_error("Error reading CSV: missing species label");
        }
        if (label_map.find(value) == label_map.end()) {
            label_map[value] = label_counter++;
            label_names.push_back(value);
        }
        labels.push_back(label_map[value]);

        dataset.push_back(data_point);
    }

    file.close();
    return dataset;
}


double random_double(double min, double max);
// Neural Network class
class NeuralNetwork {
public:
    double weights[4][3];
    double bias[3];
    double learning_rate;

    NeuralNetwork() {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                weights[i][j] = random_double(-0.5, 0.5);
            }
        }
        for (int j = 0; j < 3; ++j) {
            bias[j] = random_double(-0.5, 0.5);
        }
        learning_rate = 0.01;
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoid_derivative(double x) {
        return x * (1.0 - x);
    }

    std::vector<double> predict(const std::vector<double> &inputs) {
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

    void train(std::vector<std::vector<double>> &X, std::vector<int> &y, int epochs) {
        std::cout << BLUE << BOLD << "\nTraining Progress:\n" << RESET << std::endl;

        int bar_width = 50;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double epoch_loss = 0.0;
            for (size_t i = 0; i < X.size(); ++i) {
                std::vector<double> outputs = predict(X[i]);
                std::vector<double> target(3, 0.0);
                target[y[i]] = 1.0;

                std::vector<double> errors(3, 0.0);
                for (int j = 0; j < 3; ++j) {
                    errors[j] = target[j] - outputs[j];
                    epoch_loss += errors[j] * errors[j];
                }

                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 4; ++k) {
                        weights[k][j] += learning_rate * errors[j] * sigmoid_derivative(outputs[j]) * X[i][k];
                    }
                    bias[j] += learning_rate * errors[j] * sigmoid_derivative(outputs[j]);
                }
            }

            if ((epoch + 1) % (epochs / 100) == 0 || epoch == epochs - 1) {
                float progress = static_cast<float>(epoch + 1) / epochs;
                int pos = static_cast<int>(bar_width * progress);

                std::cout << "[";
                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << int(progress * 100.0) << "% ";
                std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << std::fixed << std::setprecision(4) << epoch_loss / X.size() << "\r";
                std::cout.flush();
            }
        }
        std::cout << std::endl;
    }

    void save_model(const std::string &filename) {
        std::ofstream file(filename);
        file << std::fixed << std::setprecision(6);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                file << weights[i][j] << " ";
            }
        }
        for (int j = 0; j < 3; ++j) {
            file << bias[j] << " ";
        }
        file.close();
    }
};

double random_double(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

void print_confusion_matrix(const std::vector<std::vector<int>>& confusion_matrix, const std::vector<std::string>& label_names) {
    std::cout << CYAN << BOLD << "\nConfusion Matrix:\n" << RESET << std::endl;
    
    // Calculate max width for labels
    size_t max_width = std::max_element(label_names.begin(), label_names.end(),
        [](const std::string& a, const std::string& b) { return a.length() < b.length(); })->length();
    max_width = std::max(max_width, size_t(15));  // Minimum width of 15

    // Print header
    std::cout << std::setw(max_width + 2) << "Predicted >";
    for (const auto &name : label_names) {
        std::cout << std::setw(max_width + 2) << name;
    }
    std::cout << std::endl;

    // Print rows
    for (size_t i = 0; i < confusion_matrix.size(); ++i) {
        std::cout << std::setw(max_width + 2) << label_names[i];
        for (size_t j = 0; j < confusion_matrix[i].size(); ++j) {
            std::cout << std::setw(max_width + 2) << confusion_matrix[i][j];
        }
        std::cout << std::endl;
    }
}

void print_metrics(const std::vector<std::vector<int>>& confusion_matrix, const std::vector<std::string>& label_names) {
    std::cout << MAGENTA << BOLD << "\nPrecision, Recall, and F1-score:\n" << RESET << std::endl;
    
    for (size_t i = 0; i < label_names.size(); ++i) {
        int true_positive = confusion_matrix[i][i];
        int false_positive = 0;
        int false_negative = 0;

        for (size_t j = 0; j < confusion_matrix.size(); ++j) {
            if (i != j) {
                false_positive += confusion_matrix[j][i];
                false_negative += confusion_matrix[i][j];
            }
        }

        double precision = true_positive / static_cast<double>(true_positive + false_positive);
        double recall = true_positive / static_cast<double>(true_positive + false_negative);
        double f1_score = 2 * (precision * recall) / (precision + recall);

        std::cout << BOLD << label_names[i] << ":" << RESET << std::endl;
        std::cout << "  Precision: " << std::fixed << std::setprecision(2) << precision << std::endl;
        std::cout << "  Recall:    " << std::fixed << std::setprecision(2) << recall << std::endl;
        std::cout << "  F1-score:  " << std::fixed << std::setprecision(2) << f1_score << std::endl;
        std::cout << std::endl;
    }
}

void evaluate_model(NeuralNetwork &nn, std::vector<std::vector<double>> &X, std::vector<int> &y, const std::vector<std::string> &label_names) {
    int correct = 0;
    std::vector<std::vector<int>> confusion_matrix(3, std::vector<int>(3, 0));

    std::cout << BLUE << BOLD << "\nEvaluating the model on the test set:" << RESET << std::endl;

    int progress = 0;
    int bar_width = 50;
    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<double> prediction = nn.predict(X[i]);
        int predicted_class = std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()));
        
        if (predicted_class == y[i]) {
            correct++;
        }
        
        confusion_matrix[y[i]][predicted_class]++;

        // Update progress bar
        int new_progress = static_cast<int>((i + 1) * 100 / X.size());
        if (new_progress > progress) {
            progress = new_progress;
            int pos = bar_width * progress / 100;
            std::cout << "[";
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << progress << "%\r";
            std::cout.flush();
        }
    }
    std::cout << std::endl;

    double accuracy = static_cast<double>(correct) / X.size();
    std::cout << GREEN << BOLD << "\nAccuracy: " << std::fixed << std::setprecision(2) << accuracy * 100 << "%" << RESET << std::endl;

    print_confusion_matrix(confusion_matrix, label_names);
    print_metrics(confusion_matrix, label_names);
}

int main() {
    try {
        std::cout << MAGENTA << BOLD << "\n======== Iris Classification with Neural Network ========" << RESET << std::endl;
        std::cout << CYAN << "Loading dataset..." << RESET << std::endl;

        std::vector<int> labels;
        std::vector<std::string> label_names;
        std::vector<std::vector<double>> dataset = read_dataset("Iris.csv", labels, label_names);

        DEBUG("Dataset loaded. Size: " << dataset.size() << " samples, " << dataset[0].size() << " features");
        DEBUG("Number of labels: " << labels.size());
        DEBUG("Label names: ");
        for (const auto& name : label_names) {
            DEBUG(" - " << name);
        }

        std::vector<size_t> indices(dataset.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        size_t train_size = dataset.size() * 0.8;
        std::vector<std::vector<double>> X_train, X_test;
        std::vector<int> y_train, y_test;

        for (size_t i = 0; i < dataset.size(); ++i) {
            if (i < train_size) {
                X_train.push_back(dataset[indices[i]]);
                y_train.push_back(labels[indices[i]]);
            } else {
                X_test.push_back(dataset[indices[i]]);
                y_test.push_back(labels[indices[i]]);
            }
        }

        DEBUG("Train set size: " << X_train.size());
        DEBUG("Test set size: " << X_test.size());

        std::cout << BLUE << BOLD << "\nInitializing Neural Network..." << RESET << std::endl;
        NeuralNetwork nn;

        std::cout << BLUE << BOLD << "\nStarting Training Process" << RESET << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        nn.train(X_train, y_train, 1000);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << GREEN << BOLD << "\nTraining completed in " << elapsed.count() << " seconds." << RESET << std::endl;

        evaluate_model(nn, X_test, y_test, label_names);

        nn.save_model("model.txt");
        std::cout << GREEN << BOLD << "\nModel trained and saved successfully!" << RESET << std::endl;

    } catch (const std::exception& e) {
        std::cerr << RED << BOLD << "An error occurred: " << e.what() << RESET << std::endl;
        return 1;
    }

    return 0;
}