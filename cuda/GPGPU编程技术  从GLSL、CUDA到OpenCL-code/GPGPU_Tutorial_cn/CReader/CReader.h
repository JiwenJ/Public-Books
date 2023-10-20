/*
 * @brief Text file reader
 * @author Deyuan Qiu
 * @date May 8, 2009
 * @file CReader.h
 */

#ifndef READER_CPP_
#define READER_CPP_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class CReader{
public:
	CReader(void){init();};

	/*
	 * Read from a text file.
	 * @param The text file name.
	 * @return Content of the file.
	 */
	char *textFileRead(char *chFileName);

private:
	void init(void);
	FILE *_fp;
	char *_content;
	int _count;
};

#endif /* READER_CPP_ */
