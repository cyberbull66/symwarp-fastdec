#pragma once

#include <string>
#include <exception>

class AppException : public std::exception {
public:
	AppException() {}

	AppException(const char *what) : m_what(what) {}

	AppException(const std::string what) : m_what(what) {}

	AppException(const AppException &that) { *this = that; }

	AppException& operator=(const AppException &that)
	{
		if (this != &that)
		{
			std::exception::operator=(that);
			m_what = that.m_what;
		}
		return *this;
	}

	const char* what() const throw() override
	{
		return m_what.c_str();
	}

private:
	std::string m_what;
};
