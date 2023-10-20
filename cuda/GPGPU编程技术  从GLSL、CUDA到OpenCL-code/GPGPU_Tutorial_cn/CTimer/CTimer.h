/*
 * @brief CPU timer for Unix
 * @author Deyuan Qiu
 * @date May 6, 2009
 * @file timer.h
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <sys/time.h>
#include <stdlib.h>

class CTimer{
public:
	CTimer(void){init();};

	/*
	 * Get elapsed time from last reset()
	 * or class construction.
	 * @return The elapsed time.
	 */
	long getTime(void);

	/*
	 * Reset the timer.
	 */
	void reset(void);

private:
	timeval _time;
	long _lStart;
	long _lStop;
	void init(void);
};

#endif /* TIMER_H_ */
