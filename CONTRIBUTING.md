# Contributing to WhisperKit

## Overview

We welcome and encourage contributions to WhisperKit! Whether you're fixing bugs, improving documentation, or adding new features from the roadmap, your help is appreciated. This guide will help you get started with contributing to WhisperKit.

## Getting Started

1. **Fork the Repository**: Start by [forking](https://github.com/argmaxinc/WhisperKit/fork) the WhisperKit repository on GitHub to your personal account.

2. **Clone Your Fork**: Clone your fork to your local machine to start making changes.

   ```bash
   git clone https://github.com/[your-username]/whisperkit.git
   cd whisperkit
   ```

## Setting Up Your Development Environment

1. **Install Dependencies**: Use the provided `Makefile` to set up your environment. Run `make setup` to install necessary dependencies.

   ```bash
   make setup
   ```

2. **Download Models**: Run `make download-models` to download the required models to run and test locally.

   ```bash
   make download-model MODEL=tiny 
   ```

## Making Changes

1. **Make Your Changes**: Implement your changes, add new features, or fix bugs. Ensure you adhere to the existing coding style. If you're adding new features, make sure to update or add any documentation or tests as needed.

2. **Build and Test**: You can use the `Makefile` to build and test your changes. Run `make build` to build WhisperKit and `make test` to run tests.

   ```bash
   make build
   make test
   ```

    You can also run and test directly from Xcode. We've provided an example app that contains various use cases, just open the `Examples/WhisperAX/WhisperAX.xcodeproj` file in Xcode and run the app.

## Submitting Your Changes

1. **Commit Your Changes**: Once you're satisfied with your changes, commit them with a clear and concise commit message.

   ```bash
   git commit -am "Add a new feature"
   ```

2. **Push to Your Fork**: Push your changes to your fork on GitHub.

   ```bash
   git push origin my-branch
   ```

3. **Create a Pull Request**: Go to the WhisperKit repository on GitHub and create a new pull request from your fork. Ensure your pull request has a clear title and description.

4. **Code Review**: Wait for the maintainers to review your pull request. Be responsive to feedback and make any necessary changes.

## Guidelines

- **Code Style**: Follow the existing code style in the project.
- **Commit Messages**: Write meaningful commit messages that clearly describe the changes.
- **Documentation**: Update documentation if you're adding new features or making changes that affect how users interact with WhisperKit.
- **Tests**: Add or update tests for new features or bug fixes.

## Final Steps

After your pull request has been reviewed and approved, a maintainer will merge it into the main branch. Congratulations, you've successfully contributed to WhisperKit!

Thank you for making WhisperKit better for everyone! ❤️‍🔥
