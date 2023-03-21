#include <chrono>
#include <iostream>
#include <random>

#include "multi_add.h"

// using std::chrono library to check time cost
using namespace std::chrono;

// size of array
const size_t size = 8192;

int main() {
	// allocate spaces with c++ method
	double *a = new double[size];
	double *b = new double[size];
	double *c = new double[size];
	double *d = new double[size];

    // generate random float numbers with c++ method
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<> distribution(0.999, 1.001);
	for (size_t i = 0; i < size; ++i) {
		a[i] = distribution(generator);
		b[i] = distribution(generator);
		c[i] = distribution(generator);
		// give initialize values for check
		d[i] = -10.0;
	}

	// record the start time with c++ method
	auto start = steady_clock::now();

	// 把用来浪费时间的循环 100,000 次放到了 cuda 的函数内部，避免多次重复 malloc 和 free 显存
	MultiAdd(a, b, c, d, size);

	// record the stop time with c++ method
	auto stop = steady_clock::now();
	// calculate the duration
	std::cout << "Cost " << duration_cast<milliseconds>(stop - start).count() << " ms.\n";

	// check result
	std::cout << d[0] << " " << d[1] << " " << d[2] << "\n";

    // free spaces with c++ method
	delete a;
	delete b;
	delete c;
	delete d;

	return 0;
}
