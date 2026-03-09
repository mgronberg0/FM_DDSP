#include"../engine/FMEngine.h"
#include <cstdio>
#include <cstdint>
#include <cstring>

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

main()
{

}
