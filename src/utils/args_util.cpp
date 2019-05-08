#include "args_util.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


void ArgsUtil::del_arg(int argc, char **argv, int index)
{
	int i;
	for (i = index; i < argc - 1; ++i) argv[i] = argv[i + 1];
	argv[i] = 0;
}

int ArgsUtil::find_arg(int argc, char* argv[], char *arg)
{
	int i;
	for (i = 0; i < argc; ++i) {
		if (!argv[i]) continue;
		if (0 == strcmp(argv[i], arg))
		{
			del_arg(argc, argv, i);
			return 1;
		}
	}
	return 0;
}

int ArgsUtil::find_int_arg(int argc, char **argv, char *arg, int def)
{
	int i;
	for (i = 0; i < argc - 1; ++i)
	{
		if (!argv[i]) continue;
		if (0 == strcmp(argv[i], arg))
		{
			def = atoi(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

float ArgsUtil::find_float_arg(int argc, char **argv, char *arg, float def)
{
	int i;
	for (i = 0; i < argc - 1; ++i)
	{
		if (!argv[i]) continue;
		if (0 == strcmp(argv[i], arg))
		{
			def = (float)atof(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

char * ArgsUtil::find_char_arg(int argc, char **argv, char *arg, char *def)
{
	int i;
	for (i = 0; i < argc - 1; ++i)
	{
		if (!argv[i]) continue;
		if (0 == strcmp(argv[i], arg))
		{
			def = argv[i + 1];
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}
