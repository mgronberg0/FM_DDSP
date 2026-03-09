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

void FMVoice::processBlock(float* buf, int numSamples)
{
    for (int i = 0; i<numSamples; i++){
        buf[i] = processSample();
    }
}

void FMVoice::setSampleRate(float Fs)
{
    Fs_ = Fs;
    for (int i = 0; i< kNumOperators; i++){
        ops_[i].env.setSampleRate(Fs);
    }
    ampEnv_.setSampleRate(Fs);
}

// Unpacks FMPatch object into the FMVoice
void FMVoice::setPatch(const FMPatch& patch)
{
    ampEnv_ = patch.ampEnv;
    algorithmIndex_ = patch.algorithmIndex;
    alg_ = getAlgorithm(algorithmIndex_);
    //TODO: where does pitchbend range get assigned
    // = patch.pitchBendRange;
    // = patch.name;
    for (int i = 0; i<kNumOperators; i++){
        ops_[i] = patch.ops[i];
    }
    // make sure all envelopes are using the patch Sample Rate
    setSampleRate(Fs_);
}

void FMVoice::noteOn(float freqHz, float velocity)
{
    fundamentalHz_ = fm_clamp(freqHz, 10.0f, 15000.0f);
    velocity_ = fm_clamp(velocity, 0.0f, 1.0f);
    for (int i = 0; i<kNumOperators; i++){
        ops_[i].env.noteOn();
    }
    ampEnv_.noteOn();
}

void FMVoice::noteOff()
{
    ampEnv_.noteOff();
}

bool FMVoice::isActive() const
{
    if (ampEnv_.state == EnvADSR::State::Idle){
        return false;
    }
    return true;
}

void VoiceAllocator::reset()
{
    for(int i = 0; i<kNumVoices; i++){
        states_[i] = {};
    }
}

void VoiceAllocator::tick()
{
    for(int i = 0; i<kNumVoices; i++){
        if (states_[i].active) states_[i].age++;
    }
    // TODO: where do we signal to the VoiceAllocator that the voice is idle
}

int VoiceAllocator::allocate(float freqHz)
{
    int oldestIndex = 0;
    int oldestAge = 0;
    for(int i = 0; i<kNumVoices; i++){
        // give up idle voices
        if (states_[i].active==false){
            states_[i].active = true;
            states_[i].freqHz = freqHz;
            states_[i].age = 0;
            return i;
        }
        // give up same voice
        if (states_[i].freqHz == freqHz){
            states_[i].active = true;
            states_[i].freqHz = freqHz;
            states_[i].age = 0;
            return i;
        }
        // give up oldest voice
        if (states_[i].age>oldestAge){
            oldestIndex = i;
            oldestAge = states_[i].age;
        }
    }
    states_[oldestIndex].active = true;
    states_[oldestIndex].freqHz = freqHz;
    states_[oldestIndex].age = 0;
    return oldestIndex;
}


