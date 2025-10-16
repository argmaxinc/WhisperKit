
<div align="center">
  
<a href="https://github.com/argmaxinc/WhisperKit#gh-light-mode-only">
  <img src="https://github.com/user-attachments/assets/f0699c07-c29f-45b6-a9c6-f6d491b8f791" alt="WhisperKit" width="20%" />
</a>

<a href="https://github.com/argmaxinc/WhisperKit#gh-dark-mode-only">
  <img src="https://github.com/user-attachments/assets/1be5e31c-de42-40ab-9b85-790cb911ed47" alt="WhisperKit" width="20%" />
</a>

# WhisperKit

[![Tests](https://github.com/argmaxinc/whisperkit/actions/workflows/release-tests.yml/badge.svg)](https://github.com/argmaxinc/whisperkit/actions/workflows/release-tests.yml)
[![License](https://img.shields.io/github/license/argmaxinc/whisperkit?logo=github&logoColor=969da4&label=License&labelColor=353a41&color=32d058)](LICENSE.md)
[![Supported Swift Version](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fargmaxinc%2FWhisperKit%2Fbadge%3Ftype%3Dswift-versions&labelColor=353a41&color=32d058)](https://swiftpackageindex.com/argmaxinc/WhisperKit) [![Supported Platforms](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fargmaxinc%2FWhisperKit%2Fbadge%3Ftype%3Dplatforms&labelColor=353a41&color=32d058)](https://swiftpackageindex.com/argmaxinc/WhisperKit)
[![Discord](https://img.shields.io/discord/1171912382512115722?style=flat&logo=discord&logoColor=969da4&label=Discord&labelColor=353a41&color=32d058&link=https%3A%2F%2Fdiscord.gg%2FG5F5GZGecC)](https://discord.gg/G5F5GZGecC)


</div>

WhisperKit is an [Argmax](https://www.takeargmax.com) framework for deploying state-of-the-art speech-to-text systems (e.g. [Whisper](https://github.com/openai/whisper)) on device with advanced features such as real-time streaming, word timestamps, voice activity detection, and more.

[[TestFlight Demo App]](https://testflight.apple.com/join/Q1cywTJw) [[Python Tools]](https://github.com/argmaxinc/whisperkittools) [[Benchmarks & Device Support]](https://huggingface.co/spaces/argmaxinc/whisperkit-benchmarks) [[WhisperKit Android]](https://github.com/argmaxinc/WhisperKitAndroid)

> [!IMPORTANT]
> WhisperKit is ideal for getting started with on-device speech-to-text. When you are ready to scale your on-device deployment with real-time transcription and speaker diarization, start your [14-day trial](https://app.argmaxinc.com) for [Argmax Pro SDK](https://www.argmaxinc.com/#SDK) with 9x faster and higher accuracy models such as Nvidia Parakeet V3, [pyannoteAI's flagship](https://www.argmaxinc.com/blog/pyannote-argmax) speaker diarization model, and a Deepgram-compatible WebSocket [local server](https://www.argmaxinc.com/blog/argmax-local-server) for easy integration into non-Swift projects.

## Table of Contents

- [Installation](#installation)
  - [Swift Package Manager](#swift-package-manager)
  - [Prerequisites](#prerequisites)
  - [Xcode Steps](#xcode-steps)
  - [Package.swift](#packageswift)
  - [Homebrew](#homebrew)
- [Getting Started](#getting-started)
  - [Quick Example](#quick-example)
  - [Model Selection](#model-selection)
  - [Generating Models](#generating-models)
  - [Swift CLI](#swift-cli)
- [WhisperKit Local Server](#whisperkit-local-server)
- [Contributing \& Roadmap](#contributing--roadmap)
- [License](#license)
- [Citation](#citation)

## Installation

### Swift Package Manager

WhisperKit can be integrated into your Swift project using the Swift Package Manager.

### Prerequisites

- macOS 14.0 or later.
- Xcode 15.0 or later.

### Xcode Steps

1. Open your Swift project in Xcode.
2. Navigate to `File` > `Add Package Dependencies...`.
3. Enter the package repository URL: `https://github.com/argmaxinc/whisperkit`.
4. Choose the version range or specific version.
5. Click `Finish` to add WhisperKit to your project.

### Package.swift

If you're using WhisperKit as part of a swift package, you can include it in your Package.swift dependencies as follows:

```swift
dependencies: [
    .package(url: "https://github.com/argmaxinc/WhisperKit.git", from: "0.9.0"),
],
```

Then add `WhisperKit` as a dependency for your target:

```swift
.target(
    name: "YourApp",
    dependencies: ["WhisperKit"]
),
```

### Homebrew

You can install `WhisperKit` command line app using [Homebrew](https://brew.sh) by running the following command:

```bash
brew install whisperkit-cli
```  

## Getting Started

To get started with WhisperKit, you need to initialize it in your project.

### Quick Example

This example demonstrates how to transcribe a local audio file:

```swift
import WhisperKit

// Initialize WhisperKit with default settings
Task {
   let pipe = try? await WhisperKit()
   let transcription = try? await pipe!.transcribe(audioPath: "path/to/your/audio.{wav,mp3,m4a,flac}")?.text
    print(transcription)
}
```

### Model Selection

WhisperKit automatically downloads the recommended model for the device if not specified. You can also select a specific model by passing in the model name:

```swift
let pipe = try? await WhisperKit(WhisperKitConfig(model: "large-v3"))
```

This method also supports glob search, so you can use wildcards to select a model:

```swift
let pipe = try? await WhisperKit(WhisperKitConfig(model: "distil*large-v3"))
```

Note that the model search must return a single model from the source repo, otherwise an error will be thrown.

For a list of available models, see our [HuggingFace repo](https://huggingface.co/argmaxinc/whisperkit-coreml).

### Generating Models

WhisperKit also comes with the supporting repo [`whisperkittools`](https://github.com/argmaxinc/whisperkittools) which lets you create and deploy your own fine tuned versions of Whisper in CoreML format to HuggingFace. Once generated, they can be loaded by simply changing the repo name to the one used to upload the model:

```swift
let config = WhisperKitConfig(model: "large-v3", modelRepo: "username/your-model-repo")
let pipe = try? await WhisperKit(config)
```

### Swift CLI

The Swift CLI allows for quick testing and debugging outside of an Xcode project. To install it, run the following:

```bash
git clone https://github.com/argmaxinc/whisperkit.git
cd whisperkit
```

Then, setup the environment and download your desired model.

```bash
make setup
make download-model MODEL=large-v3
```

**Note**:

1. This will download only the model specified by `MODEL` (see what's available in our [HuggingFace repo](https://huggingface.co/argmaxinc/whisperkit-coreml), where we use the prefix `openai_whisper-{MODEL}`)
2. Before running `download-model`, make sure [git-lfs](https://git-lfs.com) is installed

If you would like download all available models to your local folder, use this command instead:

```bash
make download-models
```

You can then run them via the CLI with:

```bash
swift run whisperkit-cli transcribe --model-path "Models/whisperkit-coreml/openai_whisper-large-v3" --audio-path "path/to/your/audio.{wav,mp3,m4a,flac}" 
```

Which should print a transcription of the audio file. If you would like to stream the audio directly from a microphone, use:

```bash
swift run whisperkit-cli transcribe --model-path "Models/whisperkit-coreml/openai_whisper-large-v3" --stream
```

### WhisperKit Local Server

WhisperKit includes a local server that implements the OpenAI Audio API, allowing you to use existing OpenAI SDK clients or generate new ones. The server supports transcription and translation with **output streaming** capabilities (real-time transcription results as they're generated).

> [!NOTE]
> **For real-time transcription server with full-duplex streaming capabilities**, check out [WhisperKit Pro Local Server](https://www.argmaxinc.com/blog/argmax-local-server) which provides live audio streaming and real-time transcription for applications requiring continuous audio processing.

#### Building the Server

```bash
# Build with server support
make build-local-server

# Or manually with the build flag
BUILD_ALL=1 swift build --product whisperkit-cli
```

#### Starting the Server

```bash
# Start server with default settings
BUILD_ALL=1 swift run whisperkit-cli serve

# Custom host and port
BUILD_ALL=1 swift run whisperkit-cli serve --host 0.0.0.0 --port 8080

# With specific model and verbose logging
BUILD_ALL=1 swift run whisperkit-cli serve --model tiny --verbose

# See all configurable parameters
BUILD_ALL=1 swift run whisperkit-cli serve --help
```

#### API Endpoints

- **POST** `/v1/audio/transcriptions` - Transcribe audio to text
- **POST** `/v1/audio/translations` - Translate audio to English

#### Supported Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `file` | Audio file (wav, mp3, m4a, flac) | Required |
| `model` | Model identifier | Server default |
| `language` | Source language code | Auto-detect |
| `prompt` | Text to guide transcription | None |
| `response_format` | Output format (json, verbose_json) | verbose_json |
| `temperature` | Sampling temperature (0.0-1.0) | 0.0 |
| `timestamp_granularities[]` | Timing detail (word, segment) | segment |
| `stream` | Enable streaming | false |

#### Client Examples

**Python Client (OpenAI SDK)**
```bash
cd Examples/ServeCLIClient/Python
uv sync
python whisperkit_client.py transcribe --file audio.wav --language en
python whisperkit_client.py translate --file audio.wav
```

Quick Python example:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:50060/v1")
result = client.audio.transcriptions.create(
    file=open("audio.wav", "rb"),
    model="tiny"  # Model parameter is required
)
print(result.text)
```

**Swift Client (Generated from OpenAPI Spec, see ServeCLIClient/Swift/updateClient.sh)**
```bash
cd Examples/ServeCLIClient/Swift
swift run whisperkit-client transcribe audio.wav --language en
swift run whisperkit-client translate audio.wav
```

**CurlClient (Shell Scripts)**
```bash
cd Examples/ServeCLIClient/Curl
chmod +x *.sh
./transcribe.sh audio.wav --language en
./translate.sh audio.wav --language es
./test.sh  # Run comprehensive test suite
```

#### Generating the API Specification

The server's OpenAPI specification and code are generated from the official OpenAI API:

```bash
# Generate latest spec and server code
make generate-server
```

#### Client Generation

You can generate clients for any language using the OpenAPI specification, for example:

```bash
# Generate Python client
swift run swift-openapi-generator generate scripts/specs/localserver_openapi.yaml \
  --output-directory python-client \
  --mode client \
  --mode types

# Generate TypeScript client
npx @openapitools/openapi-generator-cli generate \
  -i scripts/specs/localserver_openapi.yaml \
  -g typescript-fetch \
  -o typescript-client
```

#### API Limitations

Compared to the official OpenAI API, the local server has these limitations:

- **Response formats**: Only `json` and `verbose_json` supported (no plain text, SRT, VTT formats)
- **Model selection**: Client must launch server with desired model via `--model` flag

#### Fully Supported Features

The local server fully supports these OpenAI API features:

- **Include parameters**: `logprobs` parameter for detailed token-level log probabilities
- **Streaming responses**: Server-Sent Events (SSE) for real-time transcription
- **Timestamp granularities**: Both `word` and `segment` level timing
- **Language detection**: Automatic language detection or manual specification
- **Temperature control**: Sampling temperature for transcription randomness
- **Prompt text**: Text guidance for transcription style and context

## Contributing & Roadmap

Our goal is to make WhisperKit better and better over time and we'd love your help! Just search the code for "TODO" for a variety of features that are yet to be built. Please refer to our [contribution guidelines](CONTRIBUTING.md) for submitting issues, pull requests, and coding standards, where we also have a public roadmap of features we are looking forward to building in the future.

## License

WhisperKit is released under the MIT License. See [LICENSE](LICENSE) for more details.

## Citation

If you use WhisperKit for something cool or just find it useful, please drop us a note at [info@argmaxinc.com](mailto:info@argmaxinc.com)!

If you use WhisperKit for academic work, here is the BibTeX:

```bibtex
@misc{whisperkit-argmax,
   title = {WhisperKit},
   author = {Argmax, Inc.},
   year = {2024},
   URL = {https://github.com/argmaxinc/WhisperKit}
}
```
