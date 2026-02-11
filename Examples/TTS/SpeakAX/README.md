# TTS Example App

SpeakAX is the demo app for TTSKit, an on-device text-to-speech framework powered by Core ML. It runs on macOS and iOS with no server required.

## Requirements

- macOS 15.2 or later.
- iOS 18.2 or later.
- Xcode 16.3 or later.


## Building & Running

1. Open `Examples/TTS/SpeakAX/SpeakAX.xcodeproj` in Xcode.
2. Select the **SpeakAX** scheme and your target device.
3. **Set up signing:** In the project navigator, select the *SpeakAX* target. Go to the **Signing & Capabilities** tab, then:
   - Choose your Apple Developer Team from the **Team** dropdown.
   - Make sure a unique bundle identifier is set.
   - Xcode will automatically handle provisioning; resolve any signing issues if prompted.
4. Build and run (`Cmd+R`). Dependencies resolve automatically via SPM.


On first launch, the app prompts you to download a model. Downloads are cached in your Documents directory and reused across launches.

### Private HuggingFace Repos

If you use a private model fork, set a token in `ViewModel.swift`:

```swift
let config = TTSKitConfig(model: selectedPreset, token: "hf_...")
```

## Features

- **Model management.** Download, prewarm, load, and unload models from the sidebar.
- **Two model sizes.** 0.6B (fast, all platforms) and 1.7B (higher quality, style instructions, macOS only).
- **Real-time streaming.** Audio starts playing before generation completes.
- **Adaptive buffering.** The `.auto` strategy measures first-step speed and pre-buffers accordingly.
- **9 voices.** Ryan, Aiden, Ono Anna, Sohee, Eric, Dylan, Serena, Vivian, Uncle Fu.
- **10 languages.** English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.
- **Style instructions.** Natural-language prosody control (1.7B only).
- **Generation history.** All outputs saved as M4A with embedded metadata, reconstructed at launch.
- **Favorites.** Star generations to pin them at the top of the sidebar.
- **Waveform.** Live visualization that updates during streaming.
- **Settings.** Temperature, top-K, repetition penalty, max tokens, chunking, compute units.
- **Copy audio.** One-tap copy of the M4A to clipboard.

## Project Structure

```
SpeakAX/
├── SpeakAXApp.swift              entry point
├── ContentView.swift             root NavigationSplitView
├── SidebarView.swift             model management + history
├── DetailView.swift              input form, waveform, playback
├── ModelManagementView.swift     download / load / unload
├── GenerationSettingsView.swift  advanced options sheet
├── ViewModel.swift               @Observable view model
├── AudioMetadata.swift           Codable metadata embedded in .m4a files
├── WaveformView.swift            live waveform bar chart
└── ComputeUnitsView.swift        per-component compute unit picker
```

Persistence is file-based. Each generation is a self-contained `.m4a` file with all metadata (text, speaker, timings) embedded as JSON in the iTunes comment atom. No database is needed. The history is reconstructed by scanning the Documents directory at launch.

## Tests

Select the **SpeakAXTests** scheme and press `Cmd+U`.

- **Integration tests.** Download a model and run end-to-end generation. These require a network connection and take several minutes on first run.
- **Unit tests** (`AudioMetadataTests`). Round-trip metadata encoding and decoding, no network needed.
