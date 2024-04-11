.PHONY: setup setup-huggingface-cli setup-model-repo download-models download-model build build-cli test clean-package-caches

PIP_COMMAND := pip3
PYTHON_COMMAND := python3

# Define model repository and directories
MODEL_REPO := argmaxinc/whisperkit-coreml
MODEL_REPO_DIR := ./Models/whisperkit-coreml
BASE_COMPILED_DIR := ./Models


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
	@if huggingface-cli whoami; then \
		echo "Already logged in to Hugging Face."; \
	else \
		echo "Not logged in to Hugging Face."; \
		if [ -z "$$HF_TOKEN" ]; then \
			echo "Environment variable HF_TOKEN is not set. Running normal login."; \
			huggingface-cli login; \
		else \
			echo "Using HF_TOKEN from environment variable."; \
			huggingface-cli login --token $$HF_TOKEN; \
		fi; \
	fi


setup-model-repo:
	@echo "Setting up repository..."
	@mkdir -p $(BASE_COMPILED_DIR)
	@if [ -d "$(MODEL_REPO_DIR)/.git" ]; then \
		echo "Repository exists, resetting..."; \
		export GIT_LFS_SKIP_SMUDGE=1; \
		cd $(MODEL_REPO_DIR) && git fetch --all && git reset --hard origin/main && git clean -fdx; \
	else \
		echo "Repository not found, initializing..."; \
		export GIT_LFS_SKIP_SMUDGE=1; \
		git clone https://huggingface.co/$(MODEL_REPO) $(MODEL_REPO_DIR); \
	fi

# Download all models
download-models: setup-model-repo
	@echo "Downloading all models..."
	@cd $(MODEL_REPO_DIR) && \
	git lfs pull

# Download a specific model
download-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL is not set. Usage: make download-model MODEL=base"; \
		exit 1; \
	fi
	@echo "Downloading model $(MODEL)..."
	@$(MAKE) setup-model-repo
	@echo "Fetching model $(MODEL)..."
	@cd $(MODEL_REPO_DIR) && \
	git lfs pull --include="openai_whisper-$(MODEL)/*"

build:
	@echo "Building WhisperKit..."
	@swift build -v


build-cli:
	@echo "Building WhisperKit CLI..."
	@swift build -c release --product whisperkit-cli


test:
	@echo "Running tests..."
	@swift test -v

clean-package-caches:
	@trash ~/Library/Caches/org.swift.swiftpm/repositories
	@trash ~/Library/Developer/Xcode/DerivedData
