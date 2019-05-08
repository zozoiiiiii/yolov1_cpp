/************************************************************************/
/*
@author:  junliang
@brief:   
@time:    2019/02/22
*/
/************************************************************************/
#pragma once

#include <string>
#include <vector>
#include <map>

class StringUtil
{
public:
    static void splitString(std::vector<std::string>& result, const std::string& str, const std::string& delims);
    static void splitInt(std::vector<int>& result, const std::string& str, const std::string& delims);
    static void splitFloat(std::vector<float>& result, const std::string& str, const std::string& delims);
    static std::string Trim(const std::string& str, const char ch);

};
