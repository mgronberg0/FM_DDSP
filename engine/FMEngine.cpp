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

float Operator::effectiveFreq(float fundamentalHz) const
{
    return (fundamentalHz * ratio) * powf(2.0f,coarse/12.0f) *  powf(2.0,fine/1200.0f);
}

Algorithm getAlgorithm(int index)
{
    Algorithm result;
    
    switch(index)
    {
        // modulationMatrix[target][source] = true  →  source modulates target

        //   0: all carriers, no modulation          — pure additive
        case 0:
        //modulation matrix remains all zeros
        result.carrierMask[0]=1;
        result.carrierMask[1]=1;
        result.carrierMask[2]=1;
        result.carrierMask[3]=1;
        
        break;

        //   1: 0→1→2→3★                            — full linear chain
        case 1:
        result.modulationMatrix[1][0] = 1;
        result.modulationMatrix[2][1] = 1;
        result.modulationMatrix[3][2] = 1;
        result.carrierMask[3] = 1; 
        break;

        //   2: (0+1)→2→3★                          — two mods into chain
        case 2:
        result.modulationMatrix[2][0] = 1;
        result.modulationMatrix[2][1] = 1;
        result.modulationMatrix[3][2] = 1;
        result.carrierMask[3] = 1;
        break;

        //   3: 0→1★,  2→3★                         — two independent FM pairs
        case 3:
        result.modulationMatrix[1][0] = 1;
        result.modulationMatrix[3][2] = 1;
        result.carrierMask[1] = 1;
        result.carrierMask[3] = 1;
        break;

        //   4: (0+1+2)→3★                          — three mods on one carrier
        case 4:
        result.modulationMatrix[3][0] = 1;
        result.modulationMatrix[3][1] = 1;
        result.modulationMatrix[3][2] = 1;
        result.carrierMask[3] = 1;
        break;

        //   5: 0→(1★+2★+3★)                        — one mod, three carriers
        case 5:
        result.modulationMatrix[1][0] = 1;
        result.modulationMatrix[2][0] = 1;
        result.modulationMatrix[3][0] = 1;
        result.carrierMask[1] = 1;
        result.carrierMask[2] = 1;
        result.carrierMask[3] = 1;
        break;
        
        //   6: 0→1→(2★+3★)                         — chain into two carriers
        case 6:
        result.modulationMatrix[1][0] = 1;
        result.modulationMatrix[2][1] = 1;
        result.modulationMatrix[3][1] = 1;
        result.carrierMask[2] = 1;
        result.carrierMask[3] = 1;
        break;
        //   7: 0[fb]→1→2★, 3★                      — feedback chain + free carrier
        case 7:
        result.modulationMatrix[1][0] = 1;
        result.modulationMatrix[2][1] = 1;
        result.carrierMask[2] = 1;
        result.carrierMask[3] = 1;
        break;
        default:
        result.carrierMask[3] = 1;
        break;

    }
    return result;
}

float FMVoice::processSample()
{
    // Computes operators in order- assumes preset algorithms are acylcal
    // except for self-feed back on op0
    float opOutput[kNumOperators] = {};
    float opPhaseMod[kNumOperators] = {};
    float outputSample = 0.0f;
    // Compute op0 feedback (hardcoded to be the feedback op)
    opPhaseMod[0] = ops_[0].feedback*((feedbackBuf_[0]+feedbackBuf_[1])/2.0f);

    int carrierCount = 0;
    for (int opIdx = 0; opIdx<kNumOperators; opIdx++)
    {
        // solve incoming phase modulation
        // TODO: use our a priori knowledge of routing to create 
        // non-generalized modulation solution for less computation
        for (int sourceIdx=0; sourceIdx<opIdx; sourceIdx++){
            if (alg_.modulationMatrix[opIdx][sourceIdx] == true){
                opPhaseMod[opIdx] += opOutput[sourceIdx];
            }
        }
        opOutput[opIdx] = computeOperatorOutput(opIdx, opPhaseMod[opIdx]);
        if (alg_.carrierMask[opIdx] == true){
            carrierCount++;
            outputSample += opOutput[opIdx];
        }
        phase_[opIdx] += (kTwoPi * (ops_[opIdx].effectiveFreq(fundamentalHz_) / Fs_));
        phase_[opIdx] = fmodf(phase_[opIdx],kTwoPi);
    }
    
    // push new value the feedback buffer
    feedbackBuf_[0] = feedbackBuf_[1];
    feedbackBuf_[1] = opOutput[0];

    // advance ASDR VCA env with envADSR::process()
    float amp = ampEnv_.process();

    return carrierCount > 0 ? ((outputSample * amp * velocity_)  / carrierCount) : 0.0f;
    

}

float FMVoice::computeOperatorOutput(int opIdx, float phaseModulation)
{
    // y(t) = A*sin(2Pi*fc*t + I*sin(2Pi*fm*t))
    float opOutput = sinf(phase_[opIdx]+phaseModulation);
    // advances operator level ADS env with envADS::process()
    opOutput = opOutput * ops_[opIdx].env.process() * ops_[opIdx].level;
    return opOutput;
}



