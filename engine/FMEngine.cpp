#include "FMEngine.h"

void EnvADS::recalcIncrements()
{
    attackInc_ = fm_min(1.0f / fm_max(attackMs * 0.001f * Fs_, 1.0f), 1.0f);
    decayInc_ = fm_min((1.0f-endLevel) / fm_max((decayMs * 0.001f * Fs_),1.0f),1.0f);
}

void EnvADS::noteOn()
{
    recalcIncrements();
    state = State::Attack;
    if (retrigger)
    {
        currentLvl = 0.0f;
    }
}

float EnvADS::process()
{
    switch(state)
    {
        case State::Attack:
            currentLvl += attackInc_;
            if (currentLvl >= 1.0f)
            {
                currentLvl = 1.0f;
                state = State::Decay;
            }
            break;
        case State::Decay:
            currentLvl -= decayInc_;
            if (currentLvl<=endLevel)
            {
                currentLvl = endLevel;
                state = State::Hold;
            }
            break;
        case State::Hold:
            currentLvl = endLevel;
            break;
        case State::Idle:
            break;        
    }
    return currentLvl;
}

void EnvADSR::recalcIncrements()
{
    attackInc_ = fm_min(1.0f / fm_max(attackMs * 0.001f * Fs_, 1.0f), 1.0f);
    decayInc_ = fm_min((1.0f-sustainLvl) / fm_max((decayMs * 0.001f * Fs_),1.0f),1.0f);
}

void EnvADSR::noteOn()
{
    recalcIncrements();
    state = State::Attack;
    if (retrigger)
    {
        currentLvl = 0.0f;
    }
}

void EnvADSR::noteOff()
{
    releaseInc_ = fm_min(currentLvl / fm_max(releaseMs * 0.001f * Fs_, 1.0f), 1.0f);
    state = State::Release;
}

float EnvADSR::process()
{
    switch(state)
    {
        case State::Attack:
            currentLvl += attackInc_;
            if (currentLvl >= 1.0f)
            {
                currentLvl = 1.0f;
                state = State::Decay;
            }
            break;
        case State::Decay:
            currentLvl -= decayInc_;
            if (currentLvl<=sustainLvl)
            {
                currentLvl = sustainLvl;
                state = State::Sustain;
            }
            break;
        case State::Sustain:
            currentLvl = sustainLvl;
            break;
        case State::Release:
            currentLvl -= releaseInc_;
            if (currentLvl<=0.0f)
            {
                currentLvl = 0.0f;
                state = State::Idle;
            }
            break;
        case State::Idle:
            break;        
    }
    return currentLvl;
}




