/************************************************************************/
/*
@author:  junliang
@brief:   section name is not unique
@time:    2019/02/22
*/
/************************************************************************/
#pragma once

#include <string>
#include <vector>
#include <map>





class IniParser
{
    struct IniSection
    {
        std::string name;
        std::map<std::string, std::string> values;
    };
public:
    ~IniParser();
    bool LoadFromFile(const std::string& cfgfile);

    // section
    int GetSectionCount() const;
    std::string GetSectionByIndex(int index) const;

    // item
    int GetSectionItemCount(int sect_index) const;
    bool GetSectionItem(int sect_index, int item_index, std::string& key, std::string& value) const;

    int ReadInteger(size_t sect_index, const std::string& key, int def=0) const;
    std::string ReadString(size_t sect_index, const std::string& key, const std::string& def="") const;
    float ReadFloat(size_t sect_index, const std::string& key, float def=0.0f) const;

private:
    bool readLine(FILE *fp, std::string& line);
    std::string strip(const std::string& str);
private:
    std::string m_strFileName;
    std::vector<IniSection*> m_sections;
};