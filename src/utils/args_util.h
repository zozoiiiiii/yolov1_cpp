/************************************************************************/
/*
@author:  junliang
@brief:   argument from main
@time:    2019/03/22
*/
/************************************************************************/
#pragma once

class ArgsUtil
{
public:
	static void del_arg(int argc, char **argv, int index);
	static int find_arg(int argc, char* argv[], char *arg);
	static int find_int_arg(int argc, char **argv, char *arg, int def);
	static float find_float_arg(int argc, char **argv, char *arg, float def);
	static char *find_char_arg(int argc, char **argv, char *arg, char *def);
};