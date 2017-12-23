// Author: Aliaksei Mikhailiuk, 2017.

#ifndef TIMER_H
#define TIMER_H

#include <iostream> 
#include <new> 
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

/*! \class Timer
    \brief Used to time bits of code in the TrainingAlgorithm class

*/
class Timer{
private:

        //! Used to get the timing from the system
        struct timeval m_start, m_end;
        
        //! Final elapsed time 
        double m_elapsedTime;
public:

        //! Called when timing is started
        void startTiming();

        //! Called to end the timing
        double endTiming();

        //! Used to access private field of the m_elapsedTime
        double getElapsedTime() {return m_elapsedTime;};
};
 
#endif
