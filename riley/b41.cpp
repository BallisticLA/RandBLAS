#include <stdio.h>
#include <iostream>
#include <math.h>
#include <complex>

void some_function() {
    double d = 2.2;
    int i = 7;
    d = d + i; // double + int
    i = d*i; // floor(double * int)
}

void other_function() {
    auto b = true;
    auto ch = 'x';
    auto i = 123;
    auto d = 1.2;
    auto z = sqrt(d);
    std::complex<double> z2 {z, i};
    std::cout << z2 << "\n";
}

void print_square() {
    std::cout << "Enter 0, or a value you want to square.\n";

    double answer = 0.0;
    std::cin >> answer;

    if (answer != 0.0) std::cout << answer*answer << "\n";
}

bool accept3() {
    int tries = 1;
    while (tries < 4) {
        std::cout << "Do you want to proceed (y or n)?\n";
        char answer = 0;
        std::cin >> answer;

        switch (answer) {
        case 'y':
            return true;
        case 'n':
            return false;
        default:
            std::cout << "Sorry, I don't understand that.\n";
            tries += 1;
        }
    }
    return false;
}

int main() {
    some_function();
    other_function();
    print_square();
    bool res = accept3();
    std::cout << "You (perhaps implicitly) said ... " << res << "\n";
}
