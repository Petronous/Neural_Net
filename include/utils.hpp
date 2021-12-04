#pragma once

#include <Windows.h>
#include <vector>
#include <string>

#define RANDOM_SEED 7

using MyMat = Eigen::Matrix<float, -1, -1>;


void DBOut(const char* file, const int line, const std::string s)
{
	std::ostringstream os_;
	os_ << file << "(" << line << "):\n";
	os_ << s << "\n";
	OutputDebugString(os_.str().c_str());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> vec)
{
	std::string sep = "vector [";
	for (auto element : vec) {
		os << sep << element;
		sep = ", ";
	}
	os << "]\n";

	return os;
}

template<typename T>
std::string my_to_string(T& thing) {
	std::ostringstream ss;
	ss << thing;
	return ss.str();
}

float rand_float_in_range(float low, float high, int precision) {
	float x = (std::rand() % precision) / ((float)precision);
	return x * (high - low) + low;
}

template<typename T>
T clamp(T v, T lo, T hi) {
	if (v < lo) return lo;
	if (v > hi) return hi;
	return v;
}

#define DBOUT(s)       DBOut(__FILE__, __LINE__, s)