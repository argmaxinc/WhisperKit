
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
  - [Swift CLI](#swift-cli)
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
Task {
   let pipe = try? await WhisperKit()
   let transcription = try? await pipe!.transcribe(audioPath: "path/to/your/audio.{wav,mp3,m4a,flac}")?.text
    print(transcription)
}
```

### Model Selection

WhisperKit automatically downloads the recommended model for the device if not specified. You can also select a specific model by passing in the model name:

```swift
let pipe = try? await WhisperKit(model: "large-v3")
```

This method also supports glob search, so you can use wildcards to select a model:

```swift
let pipe = try? await WhisperKit(model: "distil*large-v3")
```

Note that the model search must return a single model from the source repo, otherwise an error will be thrown.

For a list of available models, see our [HuggingFace repo](https://huggingface.co/argmaxinc/whisperkit-coreml).

### Generating Models

WhisperKit also comes with the supporting repo [`whisperkittools`](https://github.com/argmaxinc/whisperkittools) which lets you create and deploy your own fine tuned versions of Whisper in CoreML format to HuggingFace. Once generated, they can be loaded by simply changing the repo name to the one used to upload the model:

```swift
let pipe = try? await WhisperKit(model: "large-v3", modelRepo: "username/your-model-repo")
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
