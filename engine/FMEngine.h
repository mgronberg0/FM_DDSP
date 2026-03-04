#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <array>

// ─────────────────────────────────────────────────────────────────────────────
// Embedded-safe utility helpers
// These replace <algorithm> to avoid any risk of pulling in heap-allocating
// STL machinery on constrained targets (Daisy Cortex-M7 etc.)
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
static constexpr T fm_min(T a, T b) { return a < b ? a : b; }

template<typename T>
static constexpr T fm_max(T a, T b) { return a > b ? a : b; }

template<typename T>
static constexpr T fm_clamp(T val, T lo, T hi) { return val < lo ? lo : (val > hi ? hi : val); }

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int   kNumOperators  = 4;
static constexpr int   kNumAlgorithms = 8;
static constexpr int   kNumVoices     = 8;
static constexpr float kTwoPi         = 6.28318530718f;

// ─────────────────────────────────────────────────────────────────────────────
// EnvADS
// A three-stage envelope: Attack, Decay, hold at endLevel indefinitely.
// Does not respond to noteOff — it has no release stage.
// Intended for per-operator modulation index shaping, where the overall
// amplitude fade is handled separately by EnvADSR on the voice.
//
// Signal flow:
//   noteOn  → ramp from currentLvl (legato) or 0 (retrigger) up to 1.0
//             over attackMs  (normalised internally)
//           → ramp from 1.0 down to endLevel over decayMs
//           → hold at endLevel until voice is killed
//   process(peakLevel) → normalisedOutput × peakLevel
//
// peakLevel (Operator::level) is applied as a live multiplier at process()
// time rather than locked in at noteOn. This means modulating Operator::level
// mid-note immediately scales the modulation index output — matching the
// Digitone behaviour where level is a live-modulatable parameter.
// The envelope itself always runs in normalised [0, 1] space internally.
//
// Defaults produce a static timbre — instant attack, no decay, full hold.
// This is the correct default for CNN encoder-matched patches (Phase 3),
// which estimate a single static spectral frame with no time-domain info.
// Phase 5 (two-frame transient/tonal analysis) will set Operator::level
// from the transient frame and endLevel from the tonal frame independently.
// ─────────────────────────────────────────────────────────────────────────────

struct EnvADS
{
    // Shape parameters
    float attackMs   = 5.0f;   // time to ramp up to peakLevel (ms)
    float decayMs    = 0.0f;   // time to ramp from peakLevel down to endLevel (ms)
    float endLevel   = 1.0f;   // level held after decay completes [0, 1]
                               // endLevel = 1.0 + decayMs = 0 → fully static timbre

    // Retrigger behaviour:
    // true  → new noteOn resets envelope to 0 and restarts attack (default)
    // false → new noteOn resumes attack from currentLvl, no discontinuity (legato)
    bool retrigger   = true;

    bool useExpCurve = true;   // exponential segments (more musical than linear)

    // State
    enum class State { Idle, Attack, Decay, Hold };

    State state      = State::Idle;
    float currentLvl = 0.0f;
    float Fs_        = 44100.0f;

    void setSampleRate(float Fs) { Fs_ = Fs; }

    // Trigger note on. Envelope shape runs in normalised [0, 1] space
    // internally — no peakLevel needed here.
    void noteOn();

    // Returns envelope output scaled by peakLevel (Operator::level).
    // peakLevel is applied as a live multiplier each sample, so modulating
    // Operator::level mid-note immediately scales the output — matching
    // the Digitone behaviour where level is a live-modulatable parameter.
    float process();

    bool isIdle() const { return state == State::Idle; }

private:
    float attackInc_ = 0.0f;
    float decayInc_  = 0.0f;

    void recalcIncrements();
};

// ─────────────────────────────────────────────────────────────────────────────
// EnvADSR
// A standard four-stage envelope: Attack, Decay, Sustain, Release.
// Responds to both noteOn and noteOff.
// Lives on FMVoice as a VCA-style amplitude gate applied to the summed
// carrier output. This is the only envelope that controls the audible
// fade-out on noteOff. The voice is not reclaimed until this reaches idle.
//
// Output contract: process() always returns a normalised scalar [0, 1].
// All amplitude scaling (velocity, master level, etc.) is applied at the
// call site in FMVoice::processSample(), not inside this struct.
// This is consistent with EnvADS — all envelopes in this engine output
// [0, 1] and scaling always happens at the call site.
//
// Signal flow:
//   noteOn  → ramp from currentLvl (legato) or 0 (retrigger) to 1.0
//             over attackMs
//           → ramp to sustainLvl over decayMs
//           → hold at sustainLvl
//   noteOff → ramp to 0 over releaseMs
//           → state → Idle  (voice allocator reclaims voice)
//
// sustainLvl is a shape parameter [0, 1] describing where within the
// envelope's own normalised range the sustain sits. It is not an external
// scaling factor — it lives here, not at the call site.
// ─────────────────────────────────────────────────────────────────────────────

struct EnvADSR
{
    // Shape parameters
    float attackMs   = 5.0f;
    float decayMs    = 50.0f;
    float sustainLvl = 0.8f;
    float releaseMs  = 200.0f;

    // Retrigger behaviour (same semantics as EnvADS)
    bool retrigger   = true;

    bool useExpCurve = true;

    // State
    enum class State { Idle, Attack, Decay, Sustain, Release };

    State state      = State::Idle;
    float currentLvl = 0.0f;
    float Fs_        = 44100.0f;

    void setSampleRate(float Fs) { Fs_ = Fs; }

    void noteOn();
    void noteOff();

    // Returns current amplitude [0, 1]. Call once per sample.
    float process();

    // Voice allocator uses this to reclaim voices after release completes
    bool isIdle() const { return state == State::Idle; }

private:
    float attackInc_  = 0.0f;
    float decayInc_   = 0.0f;
    float releaseInc_ = 0.0f;

    void recalcIncrements();
};

// ─────────────────────────────────────────────────────────────────────────────
// Operator
// One FM operator: a sinusoidal oscillator with a frequency ratio, a peak
// output level, and an EnvADS envelope shaping modulation index over time.
//
// When used as a modulator, the envelope-scaled output is added to the
// phase input of its target operator(s) — this is the modulation index.
// When used as a carrier, the envelope-scaled output feeds into the voice
// output sum (before the voice-level EnvADSR amplitude stage).
//
// level is the peak amplitude the operator reaches at the EnvADS apex.
// EnvADS::endLevel defines where it settles while the note is held.
// Together they define the operator's timbral arc:
//
//   level = 1.0, endLevel = 1.0  → static full modulation (default)
//   level = 1.0, endLevel = 0.2  → bright attack decaying to mellow sustain
//   level = 0.3, endLevel = 0.3  → static, moderate modulation
//
// CNN encoder level_head populates this field (Phase 3), representing
// the tonal/sustain character of the matched timbre.
// Phase 5 two-frame analysis will set level from the transient frame
// and EnvADS::endLevel from the tonal frame.
// ─────────────────────────────────────────────────────────────────────────────

struct Operator
{
    // Frequency
    float ratio    = 1.0f;   // freq = fundamental * ratio
    float coarse   = 0.0f;   // semitone offset (applied after ratio)
    float fine     = 0.0f;   // cent offset     (applied after coarse)

    // Peak output level [0, 1].
    // Modulation index scale when used as a modulator.
    // EnvADS drives toward this value at its apex.
    float level    = 1.0f;

    // Self-feedback [0, 1]. Only meaningful on Op 0.
    // Feeds Op 0's previous output back into its own phase input,
    // producing increasingly complex, buzzy timbres at higher values.
    float feedback = 0.0f;

    // Modulation index shaping envelope (no release stage)
    EnvADS env;

    // Returns the effective frequency given a fundamental Hz
    float effectiveFreq(float fundamentalHz) const;
};

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm
// Defines modulation routing between operators and which are carriers.
//
// modulationMatrix[target][source] = true  →  source modulates target
// carrierMask[i] = true  →  operator i contributes to audio output
//
// Index reference (operators 0–3):
//   0: all carriers, no modulation          — pure additive
//   1: 0→1→2→3★                            — full linear chain
//   2: (0+1)→2→3★                          — two mods into chain
//   3: 0→1★,  2→3★                         — two independent FM pairs
//   4: (0+1+2)→3★                          — three mods on one carrier
//   5: 0→(1★+2★+3★)                        — one mod, three carriers
//   6: 0→1→(2★+3★)                         — chain into two carriers
//   7: 0[fb]→1→2★, 3★                      — feedback chain + free carrier
// ─────────────────────────────────────────────────────────────────────────────

struct Algorithm
{
    bool modulationMatrix[kNumOperators][kNumOperators] = {};
    bool carrierMask[kNumOperators] = {};
};

// Returns the pre-defined Algorithm struct for a given index (0–7)
Algorithm getAlgorithm(int index);

// ─────────────────────────────────────────────────────────────────────────────
// FMPatch
// Complete snapshot of all parameters defining a sound.
// Serialisable to/from a C array for the Daisy patch bank.
// ─────────────────────────────────────────────────────────────────────────────

struct FMPatch
{
    Operator  ops[kNumOperators];
    EnvADSR   ampEnv;               // voice-level amplitude envelope (VCA)
    int       algorithmIndex = 1;
    float     pitchBendRange = 2.0f; // semitones
    char      name[32]       = "Init";
};

// ─────────────────────────────────────────────────────────────────────────────
// FMVoice
// A single polyphonic voice. Holds per-voice phase and envelope state.
// All DSP happens here. FMEngine owns an array of these.
//
// Per voice:
//   kNumOperators oscillators, each with an EnvADS (modulation index shaping)
//   one EnvADSR amplitude envelope (VCA — audible fade on noteOff)
//
// Voice lifecycle:
//   noteOn  → all EnvADS restart, EnvADSR restarts → voice is active
//   noteOff → EnvADS ignored, EnvADSR begins release
//   EnvADSR reaches idle → isIdle() = true → allocator reclaims voice
// ─────────────────────────────────────────────────────────────────────────────

class FMVoice
{
public:
    FMVoice() = default;

    void setSampleRate(float Fs);
    void setPatch(const FMPatch& patch);

    void noteOn (float freqHz, float velocity);  // velocity [0, 1]
    void noteOff();

    // Render one sample. Returns carrier sum × ampEnv, normalised
    // by number of active carriers.
    float processSample();

    // Render a block of samples into buf (mono, caller allocates).
    void processBlock(float* buf, int numSamples);

    // Voice stays active through the EnvADSR release tail
    bool  isActive()    const;
    bool  isIdle()      const { return !isActive(); }
    float currentNote() const { return fundamentalHz_; }

private:
    float   phase_[kNumOperators] = {};  // current phase accumulator [0, 2π)
    float   feedbackBuf_[2]       = {};  // two-sample history for op0 feedback

    EnvADS  opEnvs_[kNumOperators];      // per-operator modulation index envelopes
    EnvADSR ampEnv_;                     // voice-level VCA envelope

    // Patch state (copied in on noteOn / setPatch)
    Operator  ops_[kNumOperators];
    Algorithm alg_;
    int       algorithmIndex_ = 1;

    float fundamentalHz_ = 440.0f;
    float velocity_      = 1.0f;
    float Fs_            = 44100.0f;

    float computeOperatorOutput(int opIdx, float phaseModulation);
    float semitonesToRatio(float semitones) const;
    float centsToRatio(float cents)         const;
};

// ─────────────────────────────────────────────────────────────────────────────
// VoiceAllocator
// Tracks active voices and implements steal-oldest policy.
// ─────────────────────────────────────────────────────────────────────────────

class VoiceAllocator
{
public:
    VoiceAllocator() { reset(); }

    // Returns index of voice to use for a new note.
    // Prefers idle voices; steals the oldest active voice if all are busy.
    int allocate(float freqHz);

    // Mark a voice as released. Voice stays active until ampEnv reaches idle.
    void release(float freqHz);

    // Call each block to increment age counters
    void tick();

    void reset();

private:
    static constexpr int kNone = -1;

    struct VoiceState
    {
        bool  active = false;
        float freqHz = 0.0f;
        int   age    = 0;     // samples since noteOn, used for steal ordering
    };

    std::array<VoiceState, kNumVoices> states_;
};

// ─────────────────────────────────────────────────────────────────────────────
// FMEngine
// Top-level engine. Owns the voice pool and allocator.
// The only class that plugin / Daisy code interacts with directly.
// ─────────────────────────────────────────────────────────────────────────────

class FMEngine
{
public:
    FMEngine() = default;

    // Call before any audio processing
    void setSampleRate(float Fs);

    // Load a patch into all voices
    void loadPatch(const FMPatch& patch);

    // MIDI-style note control (MIDI note number 0–127)
    void noteOn (int midiNote, int velocity);  // velocity 0–127
    void noteOff(int midiNote);
    void allNotesOff();

    // Render stereo output. outL and outR must be pre-allocated by caller.
    // Voices are distributed slightly across the stereo field.
    void processBlock(float* outL, float* outR, int numSamples);

    // Convenience: render mono
    void processBlockMono(float* out, int numSamples);

    // Read back current patch (e.g. for plugin UI)
    const FMPatch& currentPatch() const { return patch_; }

private:
    std::array<FMVoice, kNumVoices> voices_;
    VoiceAllocator                  allocator_;
    FMPatch                         patch_;
    float                           Fs_ = 44100.0f;

    // Mild stereo spread: returns pan [-1, 1] for a given voice index
    float voicePan(int voiceIdx) const;

    static float midiNoteToFreq(int note);
};