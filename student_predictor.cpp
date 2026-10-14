#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
using namespace std;

// Beginner C++ machine-learning example:
// Multiple linear regression trained with gradient descent.
// Features: study hours, attendance percentage, previous marks.
// Target: final marks. All records below are synthetic demo data.

struct Student {
    double hours;
    double attendance;
    double previousMarks;
    double finalMarks;
};

int main() {
    vector<Student> data = {
        {1, 55, 45, 48},
        {2, 60, 50, 52},
        {2, 70, 55, 58},
        {3, 75, 60, 63},
        {4, 80, 65, 70},
        {5, 85, 70, 77},
        {6, 90, 78, 85},
        {7, 95, 82, 90},
        {1, 40, 35, 38},
        {4, 65, 58, 62},
        {6, 75, 68, 78},
        {3, 90, 72, 75}
    };

    // Normalize features so gradient descent behaves more reliably.
    // x1 = hours / 12, x2 = attendance / 100, x3 = previous marks / 100.
    double w1 = 0, w2 = 0, w3 = 0, bias = 50;
    const double learningRate = 0.03;
    const int epochs = 12000;
    const int n = static_cast<int>(data.size());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double dw1 = 0, dw2 = 0, dw3 = 0, db = 0;

        for (const Student& s : data) {
            double x1 = s.hours / 12.0;
            double x2 = s.attendance / 100.0;
            double x3 = s.previousMarks / 100.0;
            double predicted = bias + w1*x1 + w2*x2 + w3*x3;
            double error = predicted - s.finalMarks;

            dw1 += error * x1;
            dw2 += error * x2;
            dw3 += error * x3;
            db += error;
        }

        w1 -= learningRate * (2.0 / n) * dw1;
        w2 -= learningRate * (2.0 / n) * dw2;
        w3 -= learningRate * (2.0 / n) * dw3;
        bias -= learningRate * (2.0 / n) * db;
    }

    double hours, attendance, previous;
    cout << "=== SMART STUDENT PERFORMANCE PREDICTOR ===\n";
    cout << "Enter study hours per day (0-12): ";
    cin >> hours;
    cout << "Enter attendance percentage (0-100): ";
    cin >> attendance;
    cout << "Enter previous marks (0-100): ";
    cin >> previous;

    if (!cin || hours < 0 || hours > 12 ||
        attendance < 0 || attendance > 100 ||
        previous < 0 || previous > 100) {
        cout << "Invalid input. Please use the ranges shown.\n";
        return 1;
    }

    double prediction = bias + w1*(hours/12.0)
                              + w2*(attendance/100.0)
                              + w3*(previous/100.0);
    prediction = max(0.0, min(100.0, prediction));

    cout << fixed << setprecision(1);
    cout << "\nEstimated final marks: " << prediction << " / 100\n";

    if (prediction >= 80)
        cout << "Recommendation: Maintain your routine and practice past papers.\n";
    else if (prediction >= 60)
        cout << "Recommendation: Revise weekly and study in focused sessions.\n";
    else
        cout << "Recommendation: Build a daily study plan and seek help when needed.\n";

    cout << "\nModel trained on " << n << " synthetic sample records.\n";
    cout << "This is a learning demo, not a validated academic prediction tool.\n";
    return 0;
}
