.PHONY: setup setup-huggingface-cli setup-model-repo download-models download-model download-mlx-models download-mlx-model build build-cli test mlx-test clean-package-caches

PIP_COMMAND := pip3
PYTHON_COMMAND := python3

# Define model repository and directories
MODEL_REPO := argmaxinc/whisperkit-coreml
MLX_MODEL_REPO := argmaxinc/whisperkit-mlx

MODEL_REPO_DIR := ./Sources/WhisperKitTestsUtils/Models/whisperkit-coreml
MLX_MODEL_REPO_DIR := ./Sources/WhisperKitTestsUtils/Models/whisperkit-mlx
BASE_MODEL_DIR := ./Sources/WhisperKitTestsUtils/Models


setup:
	@echo "Setting up environment..."
	@which $(PIP_COMMAND)
	@which $(PYTHON_COMMAND)
	@echo "Checking for Homebrew..."
	@which brew > /dev/null || (echo "Error: Homebrew is not installed. Install it form here https://brew.sh and try again" && exit 1)
	@echo "Homebrew is installed."
	@echo "Checking for huggingface-cli..."
	@which huggingface-cli > /dev/null || (echo "Installing huggingface-cli..." && brew install huggingface-cli)
	@echo "huggingface-cli is installed."
	@echo "Checking for git-lfs..."
	@which git-lfs > /dev/null || (echo "Installing git-lfs..." && brew install git-lfs)
	@echo "git-lfs is installed."
	@echo "Checking for trash..."
	@which trash > /dev/null || (echo "Installing trash..." && brew install trash)
	@echo "trash is installed."
	@echo "Done 🚀"


setup-huggingface-cli:
	@if huggingface-cli whoami 2>&1 | grep -q "Not logged in"; then \
		echo "Not logged in to Hugging Face."; \
		if [ -z "$$HF_TOKEN" ]; then \
			echo "Environment variable HF_TOKEN is not set. Running normal login."; \
			huggingface-cli login; \
		else \
			echo "Using HF_TOKEN from environment variable."; \
			huggingface-cli login --token $$HF_TOKEN; \
		fi; \
	else \
		echo "Already logged in to Hugging Face."; \
		huggingface-cli whoami; \
	fi


setup-model-repo:
	@echo "Setting up repository..."
	@mkdir -p $(BASE_MODEL_DIR)
	@if [ -d "$(MODEL_REPO_DIR)/.git" ]; then \
		echo "Repository exists, resetting..."; \
		export GIT_LFS_SKIP_SMUDGE=1; \
		cd $(MODEL_REPO_DIR) && git fetch --all && git reset --hard origin/main && git clean -fdx; \
	else \
		echo "Repository not found, initializing..."; \
		export GIT_LFS_SKIP_SMUDGE=1; \
		git clone https://huggingface.co/$(MODEL_REPO) $(MODEL_REPO_DIR); \
	fi


setup-mlx-model-repo:
	@echo "Setting up mlx repository..."
	@mkdir -p $(BASE_MODEL_DIR)
	@if [ -d "$(MLX_MODEL_REPO_DIR)/.git" ]; then \
		echo "Repository exists, resetting..."; \
		export GIT_LFS_SKIP_SMUDGE=1; \
		cd $(MLX_MODEL_REPO_DIR) && git fetch --all && git reset --hard origin/main && git clean -fdx; \
	else \
		echo "Repository not found, initializing..."; \
		export GIT_LFS_SKIP_SMUDGE=1; \
		git clone https://huggingface.co/$(MLX_MODEL_REPO) $(MLX_MODEL_REPO_DIR); \
	fi


# Download all models
download-models: setup-model-repo
	@echo "Downloading all models..."
	@cd $(MODEL_REPO_DIR) && \
	git lfs pull
	@echo "CoreML models downloaded to $(MODEL_REPO_DIR)"


# Download a specific model
download-model: setup-model-repo
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL is not set. Usage: make download-model MODEL=tiny"; \
		exit 1; \
	fi
	@echo "Downloading model $(MODEL)..."
	@cd $(MODEL_REPO_DIR) && \
	git lfs pull --include="openai_whisper-$(MODEL)/*"
	@echo "CoreML model $(MODEL) downloaded to $(MODEL_REPO_DIR)/openai_whisper-$(MODEL)"


download-mlx-models: setup-mlx-model-repo
	@echo "Downloading all mlx models..."
	@cd $(MLX_MODEL_REPO_DIR) && \
	git lfs pull
	@echo "MLX models downloaded to $(MLX_MODEL_REPO_DIR)"


download-mlx-model: setup-mlx-model-repo
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL is not set. Usage: make download-mlx-model MODEL=tiny"; \
		exit 1; \
	fi
	@echo "Downloading mlx model $(MODEL)..."
	@cd $(MLX_MODEL_REPO_DIR) && \
	git lfs pull --include="openai_whisper-$(MODEL)/*"
	@echo "MLX model $(MODEL) downloaded to $(MLX_MODEL_REPO_DIR)/openai_whisper-mlx-$(MODEL)"


build:
	@echo "Building WhisperKit..."
	@xcodebuild CLANG_ENABLE_CODE_COVERAGE=NO VALID_ARCHS=arm64 clean build \
		-configuration Release \
		-scheme whisperkit-Package \
		-destination generic/platform=macOS \
		-derivedDataPath .build/.xcodebuild/ \
		-clonedSourcePackagesDirPath .build/ \
		-skipPackagePluginValidation


build-cli:
	@echo "Building WhisperKit CLI..."
	@xcodebuild CLANG_ENABLE_CODE_COVERAGE=NO VALID_ARCHS=arm64 clean build \
		-configuration Release \
		-scheme whisperkit-cli \
		-destination generic/platform=macOS \
		-derivedDataPath .build/.xcodebuild/ \
		-clonedSourcePackagesDirPath .build/ \
		-skipPackagePluginValidation


test:
	@echo "Running tests..."
	@xcodebuild clean build-for-testing test \
		-scheme whisperkit-Package \
		-only-testing WhisperKitMLXTests/MLXUnitTests \
		-only-testing WhisperKitTests/UnitTests \
		-destination 'platform=macOS,arch=arm64' \
		-skipPackagePluginValidation


clean-package-caches:
	@trash ~/Library/Caches/org.swift.swiftpm/repositories
	@trash ~/Library/Developer/Xcode/DerivedData
