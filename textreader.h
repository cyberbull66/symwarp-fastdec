#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "appexception.h"

class TextReader
{
public:
	static std::vector<std::string> Read(const char *txtFile)
	{
		std::vector<std::string> lines;
		std::ifstream fs;
		fs.open(txtFile);
		if (!fs.is_open())
			throw AppException((std::string("Can't open the file: ") + txtFile).c_str());
		std::string line;
		while (getline(fs, line))
		{
			lines.push_back(line);
		}
		return std::move(lines);
	}
};
