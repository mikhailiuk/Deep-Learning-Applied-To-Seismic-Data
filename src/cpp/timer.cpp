// Author: Aliaksei Mikhailiuk, 2017.

#include "../headers/Timer.h"


void Timer::startTiming(){
        gettimeofday(&m_start, NULL);
}

double Timer::endTiming(){
        gettimeofday(&m_end, NULL);
        m_elapsedTime =  ((m_end.tv_sec  - m_start.tv_sec) * 1000000u + m_end.tv_usec - m_start.tv_usec) / 1.e6;
        return m_elapsedTime;
}
