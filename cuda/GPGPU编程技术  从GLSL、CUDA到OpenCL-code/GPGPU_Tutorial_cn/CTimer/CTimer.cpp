/*
 * @brief CPU timer for Unix
 * @author Deyuan Qiu
 * @date May 6, 2009
 * @file timer.cpp
 */

#include "CTimer.h"

void CTimer::init(void){
	_lStart = 0;
	_lStop = 0;
	gettimeofday(&_time, NULL);
	_lStart = (_time.tv_sec * 1000) + (_time.tv_usec / 1000);
}

long CTimer::getTime(void){
	gettimeofday(&_time, NULL);
	_lStop = (_time.tv_sec * 1000) + (_time.tv_usec / 1000) - _lStart;

	return _lStop;
}

void CTimer::reset(void){
	init();
}
