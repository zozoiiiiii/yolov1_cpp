#include "string_util.h"


void StringUtil::splitString(std::vector<std::string>& result, const std::string & str, const std::string& delims)
{
    if (0 == str.compare(""))
    {
        return;
    }

    size_t start, pos;
    start = 0;
    do
    {
        pos = str.find(delims, start);
        if (pos == std::string::npos)
        {
            std::string tmp_str = str.substr(start);
            tmp_str = Trim(tmp_str, 10); // \n
            tmp_str = Trim(tmp_str, 13); // \r
            tmp_str = Trim(tmp_str, 32); // space
            result.push_back(tmp_str);
            break;
        }
        else
        {
            std::string tmp_str = str.substr(start, pos - start);
            tmp_str = Trim(tmp_str, 10); // 
            tmp_str = Trim(tmp_str, 13); // 
            tmp_str = Trim(tmp_str, 32); // 
            result.push_back(tmp_str);
            start = pos + delims.length();
        }

    } while (pos != std::string::npos);
    return;
}


void StringUtil::splitInt(std::vector<int>& result, const std::string & str, const std::string& delims)
{
    if (0 == str.compare(""))
    {
        return;
    }

    size_t start, pos;
    start = 0;
    do
    {
        pos = str.find(delims, start);
        if (pos == std::string::npos)
        {
            std::string tmp_str = str.substr(start);
            tmp_str = Trim(tmp_str, 10); // \n
            tmp_str = Trim(tmp_str, 13); // \r
            tmp_str = Trim(tmp_str, 32); // space
            result.push_back(atoi(tmp_str.c_str()));
            break;
        }
        else
        {
            std::string tmp_str = str.substr(start, pos - start);
            tmp_str = Trim(tmp_str, 10); // 
            tmp_str = Trim(tmp_str, 13); // 
            tmp_str = Trim(tmp_str, 32); // 
            result.push_back(atoi(tmp_str.c_str()));
            start = pos + delims.length();
        }

    } while (pos != std::string::npos);
    return;
}


void StringUtil::splitFloat(std::vector<float>& result, const std::string & str, const std::string& delims)
{
    if (0 == str.compare(""))
    {
        return;
    }

    size_t start, pos;
    start = 0;
    do
    {
        pos = str.find(delims, start);
        if (pos == std::string::npos)
        {
            std::string tmp_str = str.substr(start);
            tmp_str = Trim(tmp_str, 10); // \n
            tmp_str = Trim(tmp_str, 13); // \r
            tmp_str = Trim(tmp_str, 32); // space
            result.push_back(atof(tmp_str.c_str()));
            break;
        }
        else
        {
            std::string tmp_str = str.substr(start, pos - start);
            tmp_str = Trim(tmp_str, 10); // 
            tmp_str = Trim(tmp_str, 13); // 
            tmp_str = Trim(tmp_str, 32); // 
            result.push_back(atof(tmp_str.c_str()));
            start = pos + delims.length();
        }

    } while (pos != std::string::npos);
    return;
}

std::string StringUtil::Trim(const std::string& str, const char ch)
{
    if (str.empty())
    {
        return "";
    }

    if (str[0] != ch && str[str.size() - 1] != ch)
    {
        return str;
    }

    size_t pos_begin = str.find_first_not_of(ch, 0);
    size_t pos_end = str.find_last_not_of(ch, str.size());

    if (pos_begin == std::string::npos || pos_end == std::string::npos)
    {
        return "";
    }

    return str.substr(pos_begin, pos_end - pos_begin + 1);
}
