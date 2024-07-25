
<div align="center">
  
<a href="https://github.com/argmaxinc/WhisperKit#gh-light-mode-only">
  <img src="https://github.com/argmaxinc/WhisperKit/assets/1981179/6ac3360b-2f5c-4392-a71a-05c5dda71093" alt="WhisperKit" width="20%" />
</a>

<a href="https://github.com/argmaxinc/WhisperKit#gh-dark-mode-only">
  <img src="https://github.com/argmaxinc/WhisperKit/assets/1981179/a682ce21-80e0-4a98-a99f-836663538a4f" alt="WhisperKit" width="20%" />
</a>

# WhisperKit

[![Tests](https://github.com/argmaxinc/whisperkit/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/argmaxinc/whisperkit/actions/workflows/pre-release-tests.yml)
[![License](https://img.shields.io/github/license/argmaxinc/whisperkit?logo=github&logoColor=969da4&label=License&labelColor=353a41&color=32d058)](LICENSE.md)
[![Supported Swift Version](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fargmaxinc%2FWhisperKit%2Fbadge%3Ftype%3Dswift-versions&labelColor=353a41&color=32d058)](https://swiftpackageindex.com/argmaxinc/WhisperKit) [![Supported Platforms](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fargmaxinc%2FWhisperKit%2Fbadge%3Ftype%3Dplatforms&labelColor=353a41&color=32d058)](https://swiftpackageindex.com/argmaxinc/WhisperKit)
[![Discord](https://img.shields.io/discord/1171912382512115722?style=flat&logo=discord&logoColor=969da4&label=Discord&labelColor=353a41&color=32d058&link=https%3A%2F%2Fdiscord.gg%2FG5F5GZGecC)](https://discord.gg/G5F5GZGecC)


</div>

WhisperKit is a Swift package that integrates OpenAI's popular [Whisper](https://github.com/openai/whisper) speech recognition model with Apple's CoreML framework for efficient, local inference on Apple devices.

Check out the demo app on [TestFlight](https://testflight.apple.com/join/LPVOyJZW).

[[Blog Post]](https://www.takeargmax.com/blog/whisperkit) [[Python Tools Repo]](https://github.com/argmaxinc/whisperkittools)

## Table of Contents

- [Installation](#installation)
  - [Swift Package Manager](#swift-package-manager)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)
  - [Homebrew](#homebrew)
- [Getting Started](#getting-started)
  - [Quick Example](#quick-example)
  - [Model Selection](#model-selection)
  - [Generating Models](#generating-models)
  - [Testing](#testing)
- [Contributing \& Roadmap](#contributing--roadmap)
- [License](#license)
- [Citation](#citation)

## Installation

### Swift Package Manager

WhisperKit can be integrated into your Swift project using the Swift Package Manager.

### Prerequisites

- macOS 14.0 or later.
- Xcode 15.0 or later.

### Steps

1. Open your Swift project in Xcode.
2. Navigate to `File` > `Add Package Dependencies...`.
3. Enter the package repository URL: `https://github.com/argmaxinc/whisperkit`.
4. Choose the version range or specific version.
5. Click `Finish` to add WhisperKit to your project.

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
let pipe = try await WhisperKit()
// Transcribe the audio file
let transcription = try await pipe.transcribe(audioPath: "path/to/your/audio.{wav,mp3,m4a,flac}")?.text
// Print the transcription
print(transcription)
```

### Model Selection

You have to specify the model by passing the model name:

```swift
let pipe = try await WhisperKit(model: "large-v3")
```

This method also supports glob search, so you can use wildcards to select a model:

```swift
let pipe = try await WhisperKit(model: "distil*large-v3")
```

Note that the model search must return a single model from the source repo, otherwise an error will be thrown.

For a list of available models, see our [HuggingFace repo](https://huggingface.co/argmaxinc/whisperkit-coreml).
For MLX models, see [here](https://huggingface.co/argmaxinc/whisperkit-mlx).

If you want to get the recommended model for your device, you can use the following method:

```swift
print(WhisperKit.recommendedModel())
```

### Generating Models

WhisperKit also comes with the supporting repo [`whisperkittools`](https://github.com/argmaxinc/whisperkittools) which lets you create and deploy your own fine tuned versions of Whisper in CoreML format to HuggingFace. Once generated, they can be loaded by simply changing the repo name to the one used to upload the model:

```swift
let pipe = try await WhisperKit(model: "large-v3", modelRepo: "username/your-model-repo")
```

### Backend Selection

WhisperKit supports both CoreML and MLX backends. By default, it uses CoreML, but you can switch some or all pipeline components to MLX.
Available pipeline components are:
- `featureExtractor`, `FeatureExtractor` is used by default, use `MLXFeatureExtractor` to switch to MLX
- `audioEncoder`, `AudioEncoder` is used by default, use `MLXAudioEncoder` to switch to MLX
- `textDecoder`, `TextDecoder` is used by default, use `MLXTextDecoder` to switch to MLX

Here is an example of how to switch the `featureExtractor` and `audioEncoder` to MLX and keep the `textDecoder` as CoreML:

```swift
let pipe = try await WhisperKit(
  model: "tiny", 
  mlxModel: "tiny", 
  featureExtractor: MLXFeatureExtractor(),
  audioEncoder: MLXAudioEncoder()
)
```

### Testing

If you want to run the unit tests locally, first clone the repo:

```bash
git clone https://github.com/argmaxinc/whisperkit.git
cd whisperkit
```

download the required models:

```bash
make setup
make download-model MODEL=tiny
make download-mlx-model MODEL=tiny
```

and then run the tests:

```bash
make test
```

## Contributing & Roadmap

Our goal is to make WhisperKit better and better over time and we'd love your help! Just search the code for "TODO" for a variety of features that are yet to be built. Please refer to our [contribution guidelines](CONTRIBUTING.md) for submitting issues, pull requests, and coding standards, where we also have a public roadmap of features we are looking forward to building in the future.

## License

WhisperKit is released under the MIT License. See [LICENSE](LICENSE) for more details.

## Citation

If you use WhisperKit for something cool or just find it useful, please drop us a note at [info@takeargmax.com](mailto:info@takeargmax.com)!

If you use WhisperKit for academic work, here is the BibTeX:

```bibtex
@misc{whisperkit-argmax,
   title = {WhisperKit},
   author = {Argmax, Inc.},
   year = {2024},
   URL = {https://github.com/argmaxinc/WhisperKit}
}
```
