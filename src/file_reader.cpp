#include <string>
#include <fstream>
#include <boost/regex.hpp>
#include "cs_internal.h"

#include <iostream>

std::vector<float> cu::readFile(std::string path)
{
    std::vector<float> quads;
    boost::regex line_regex("\\[([ ]*-?[0-9]+\\.?[0-9]*[ ]*){10}\\]");
    boost::regex number_regex("-?[0-9]+\\.?[0-9]*");

    std::ifstream infile(path.c_str());

    std::string line;
    while (std::getline(infile, line))
    {
        std::vector<float> data;
        boost::smatch results;
        std::string::const_iterator begin = line.begin();
        std::string::const_iterator end = line.end();
        if (boost::regex_match(line, results, line_regex))
        {
            while (boost::regex_search(begin, end, results, number_regex))
            {
                for (int i=0;i<results.size(); ++i)
                {
                    std::stringstream ss;
                    ss << results[i];
                    float v;
                    ss >> v;
                    quads.push_back(v);
                }
                begin = results[results.size()-1].second;
            }
        }
    }

    return quads;
}
