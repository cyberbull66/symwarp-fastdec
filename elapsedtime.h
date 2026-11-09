#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include <sys/time.h>
#include <sys/resource.h>

class SystemTimer
{
private:
	double m_start_time;
	double m_first_start_time;
	double m_end_time;
	std::vector<std::pair<std::string, double>> m_time_series;
	int m_counter;
	bool m_started;

public:
	SystemTimer() : m_start_time(0.0), m_first_start_time(0.0), m_end_time(0.0), m_counter(0), m_started(false) 
	{
		m_first_start_time = get_wall_time();
	}

	void Start(bool reset = true)
	{
		assert(!m_started);
		m_started = true;
		m_start_time = get_wall_time();
		if (reset)
			m_first_start_time = m_start_time;
		m_counter++;
	}

    void Resume()
    {
        Start(false);
    }

	double Stop(const char *marker = nullptr)
	{
		assert(m_started);
		m_started = false;
		m_end_time = get_wall_time();
		double elapsedTime = m_end_time - m_start_time;
		m_time_series.push_back(std::make_pair(marker == nullptr ? ToString(m_counter) : marker, elapsedTime));
		return elapsedTime;
	}

	double EnduranceTime()
	{
		return (m_end_time - m_first_start_time);
	}

	double AccumulatedTime(const char *marker = nullptr)
	{
		double accumulatedTime = 0.0;
		for (auto &pair : m_time_series)
		{
			if (marker != nullptr && strcmp(marker, pair.first.c_str()) != 0)
				continue;
			accumulatedTime += pair.second;
		}
		return accumulatedTime;
	}

	void Print()
	{
		std::cout << "Elapsed time measurements (system time): " << std::endl;
		for (auto &pair : m_time_series)
		{
			std::cout << pair.first << ": " << pair.second << std::endl;
		}
		std::cout << "Endurance time: " << EnduranceTime() << std::endl;
		std::cout << "Accumulated time: " << AccumulatedTime() << std::endl;
	}

private:
	double get_wall_time(){
		struct timeval time;
		if (gettimeofday(&time, nullptr))
		{
			//  Handle error
			assert(false);
			return 0;
		}
		return (double)time.tv_sec + (double)time.tv_usec * .000001;
	}

	std::string ToString(int n)
	{
		std::stringstream ss;
		ss << n;
		return ss.str();
	}
};

class CpuTimer
{
private:
	rusage m_start_time;
	rusage m_first_start_time;
	rusage m_end_time;
	std::vector<std::pair<std::string, double>> m_time_series;
	int m_counter;
	bool m_started;

public:
	CpuTimer() : m_counter(0), m_started(false) 
	{
		getrusage(RUSAGE_SELF, &m_first_start_time);
	}

	void Start(bool reset = true)
	{
		assert(!m_started);
		m_started = true;
		getrusage(RUSAGE_SELF, &m_start_time);
		if (reset)
			m_first_start_time = m_start_time;
		m_counter++;
	}

    void Resume()
    {
        Start(false);
    }

	double Stop(const char *marker = nullptr)
	{
		assert(m_started);
		m_started = false;
		getrusage(RUSAGE_SELF, &m_end_time);
		double elapsedTime = elapsed_time(&m_start_time, &m_end_time);
		m_time_series.push_back(std::make_pair(marker == nullptr ? ToString(m_counter) : marker, elapsedTime));
		return elapsedTime;
	}

	double EnduranceTime()
	{
		return elapsed_time(&m_first_start_time, &m_end_time);
	}

	double AccumulatedTime()
	{
		double accumulatedTime = 0.0;
		for (auto &pair : m_time_series)
		{
			accumulatedTime += pair.second;
		}
		return accumulatedTime;
	}

	void Print()
	{
		std::cout << "Elapsed time measurements (CPU time): " << std::endl;
		for (auto &pair : m_time_series)
		{
			std::cout << pair.first << ": " << pair.second << std::endl;
		}
		std::cout << "Endurance time: " << EnduranceTime() << std::endl;
		std::cout << "Accumulated time: " << AccumulatedTime() << std::endl;
	}

private:
	double elapsed_time(rusage *start, rusage *end) 
	{
		timeval diff;
		timeval_subtract(&diff, &end->ru_utime, &start->ru_utime);
		return (double)diff.tv_sec + (double)diff.tv_usec / 1e6;
	}

	int timeval_subtract(timeval *result, timeval *x, timeval *y) 
	{
		/* Perform the carry for the later subtraction by updating y. */
		if (x->tv_usec < y->tv_usec) 
		{
			int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
			y->tv_usec -= 1000000 * nsec;
			y->tv_sec += nsec;
		}
		if (x->tv_usec - y->tv_usec > 1000000) 
		{
			int nsec = (x->tv_usec - y->tv_usec) / 1000000;
			y->tv_usec += 1000000 * nsec;
			y->tv_sec -= nsec;
		}

		result->tv_sec = x->tv_sec - y->tv_sec;
		result->tv_usec = x->tv_usec - y->tv_usec;

		return x->tv_sec < y->tv_sec;
	}

	std::string ToString(int n)
	{
		std::stringstream ss;
		ss << n;
		return ss.str();
	}
};
