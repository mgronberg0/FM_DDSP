#include"../engine/FMEngine.h"
#include <cstdio>
#include <cstdint>
#include <cstring>

static constexpr uint16_t bufferSize = 128;

struct MidiEvent
{
    uint32_t sampleTime; // when, in samples, this event takes place
    int midiNote;
    int velocity;  // 0 means note off, anything else is note on
}

void writeWavHeader(FILE* file, uint32_t sampleRate, uint32_t numSamples, uint16_t numChannels)
{
    uint32_t fileSize      = (numSamples * 2 * numChannels) + 36;
    uint32_t fmtChunkSize  = 16;
    uint16_t pcmFormat     = 1;
    uint32_t byteRate      = sampleRate * 2 * numChannels;
    uint16_t blockAlign    = numChannels * 2;
    uint16_t bitsPerSample = 16;
    uint32_t sampleDataSize = numSamples * 2 * numChannels;

    fwrite("RIFF",          1,                    4, file);
    fwrite(&fileSize,       sizeof(fileSize),      1, file);
    fwrite("WAVE",          1,                    4, file);
    fwrite("fmt ",          1,                    4, file);
    fwrite(&fmtChunkSize,   sizeof(fmtChunkSize),  1, file);
    fwrite(&pcmFormat,      sizeof(pcmFormat),     1, file);
    fwrite(&numChannels,    sizeof(numChannels),   1, file);
    fwrite(&sampleRate,     sizeof(sampleRate),    1, file);
    fwrite(&byteRate,       sizeof(byteRate),      1, file);
    fwrite(&blockAlign,     sizeof(blockAlign),    1, file);
    fwrite(&bitsPerSample,  sizeof(bitsPerSample), 1, file);
    fwrite("data",          1,                    4, file);
    fwrite(&sampleDataSize, sizeof(sampleDataSize),1, file);

}

void convertFloatBufferToInt16(float *bufferFloat, int16_t *bufferInt, uint32_t numSamples)
{
    float int16ConversionFactor = 32767.0f;
    for(int n = 0; n<numSamples; n++){
        bufferInt[n] = int16_t(fm_clamp(bufferFloat[n], -1.0f, 1.0f) * int16ConversionFactor);
    }
}

void render

int algoTest(FMEngine& engine, uint32_t Fs, const char* outFilePath){
    int numAlgos = 8;
    float noteLengthSec = 0.5f;
    float renderLengthSec = 1.0f;
    float audioLengthSec = numAlgos * renderLengthSec;
    int noteNum = 41;

    uint32_t noteLengthSamp = uint32_t(noteLengthSec * Fs);
    uint32_t renderLengthSamp = uint32_t(renderLengthSec * Fs);
    uint32_t audioLengthSamp = uint32_t(renderLengthSec*Fs);
    uint32_t samplesThisBlock = 0;

    float bufferFloat[bufferSize];
    int16_t bufferInt[bufferSize];

    FMPatch patch = engine.currentPatch();

    // Create audio file
    FILE* outFile = fopen(outFilePath, "wb");
    if (outFile == nullptr){
        printf("Failed to open output file\n");
        return 1;
    }
    writeWavHeader(outFile, Fs, audioLengthSamp, 1);

    bool noteStopped = false;
    for(int i = 0; i<numAlgos; i++){
        patch.algorithmIndex = i;
        engine.loadPatch(patch);
        engine.noteOn(noteNum, 60);
        noteStopped = false;
        
    }
    return 0;
}

void allocatorTest(FMEngine& engine, uint32_t Fs, const char* outFilePath){

}

int main()
{
    uint32_t Fs = 44100;
    uint16_t numChan = 1;
    uint16_t noteNum = 41;
    float noteLengthSec = 2.0f;
    uint32_t noteLengthSamp = uint32_t(noteLengthSec * Fs);
    float audioLengthSec = 5.0f;
    uint32_t audioLengthSamp = uint32_t(audioLengthSec * Fs);

    float bufferFloat[bufferSize];
    int16_t bufferInt[bufferSize];
    

    FMEngine engine;
    engine.setSampleRate(Fs);

    FMPatch patch;
    patch.algorithmIndex= 3;

    patch.algorithmIndex = 4;  // (op0+op1+op2)→op3★
    patch.ops[0].ratio = 1.0f;
    patch.ops[0].level = 1.0f;
    patch.ops[0].env.decayMs = 600.0f;
    patch.ops[0].env.endLevel = 0.1f;
    patch.ops[0].feedback = 0.70f;
    patch.ops[1].ratio = 2.0f;
    patch.ops[1].level = 1.0f;
    patch.ops[1].env.decayMs = 500.0f;
    patch.ops[1].env.endLevel = 0.0f;
    patch.ops[2].ratio = 3.0f;
    patch.ops[2].level = 1.0f;
    patch.ops[2].env.decayMs = 600.0f;
    patch.ops[2].env.endLevel = 0.0f;
    patch.ops[3].ratio = 1.0f;
    patch.ops[3].level = 1.0f;
    patch.ampEnv.attackMs = 2.0f;
    patch.ampEnv.decayMs = 200.0f;
    patch.ampEnv.sustainLvl = 0.7f;
    patch.ampEnv.releaseMs = 100.0f;

    patch.ampEnv.decayMs = 300.0f;
    patch.ampEnv.releaseMs = 3000.0f;

    engine.loadPatch(patch);

    
    // Create audio file
    FILE* outFile = fopen("/home/marcus/FM_DDSP/testOut/test2.wav", "wb");
    if (outFile == nullptr){
        printf("Failed to open output file\n");
        return 1;
    }
    writeWavHeader(outFile, Fs, audioLengthSamp, numChan);
    engine.noteOn(noteNum, 100);
    
    bool noteStopped = false;
    uint32_t samplesThisBlock = 0;
    // populate audio file
    for(int n = 0; n<audioLengthSamp; n +=bufferSize)
    {
        samplesThisBlock = fm_min((uint32_t)bufferSize, audioLengthSamp - (uint32_t)n);
        if(!noteStopped && n>noteLengthSamp){
            engine.noteOff(noteNum);
            noteStopped = true;
        }
        engine.processBlock(bufferFloat,bufferSize);
        
        // convert to uint16
        convertFloatBufferToInt16(bufferFloat,bufferInt,bufferSize);
        fwrite(bufferInt, 2, samplesThisBlock, outFile);
    }
    fclose(outFile);

    return 0;
}


