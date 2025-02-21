.PHONY: setup setup-huggingface-cli setup-model-repo download-models download-model build build-cli test clean-package-caches list-devices benchmark-connected-devices benchmark-device benchmark-devices extract-xcresult

PIP_COMMAND := pip3
PYTHON_COMMAND := python3

# Define model repository and directories
MODEL_REPO := argmaxinc/whisperkit-coreml
MODEL_REPO_DIR := ./Models/whisperkit-coreml
BASE_COMPILED_DIR := ./Models

GIT_HASH := $(shell git rev-parse --short HEAD)

setup:
	@echo "Setting up environment..."
	@which $(PIP_COMMAND)
	@which $(PYTHON_COMMAND)
	@echo "Checking for Homebrew..."
	@which brew > /dev/null || (echo "Error: Homebrew is not installed. Install it from https://brew.sh and try again" && exit 1)
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
	@echo "Checking for fastlane"
	@which fastlane > /dev/null || (echo "Installing fastlane..." && brew install fastlane)
	@echo "fastlane is installed."
	@$(MAKE) generate-whisperax-xcconfig
	@echo "Done 🚀"


generate-whisperax-xcconfig:
	@echo "Updating DEVELOPMENT_TEAM in Examples/WhisperAX/Debug.xcconfig..."
	@TEAM_ID=$$(defaults read com.apple.dt.Xcode IDEProvisioningTeams | plutil -convert json -r -o - -- - | jq -r  'to_entries[0].value | sort_by(.teamType == "Individual") | .[0].teamID' 2>/dev/null); \
	if [ -z "$$TEAM_ID" ]; then \
		echo "Error: No Development Team ID found. Please log into Xcode with your Apple ID and select a team."; \
	else \
		echo "DEVELOPMENT_TEAM=$$TEAM_ID" > Examples/WhisperAX/Debug.xcconfig; \
		echo "DEVELOPMENT_TEAM has been updated in Examples/WhisperAX/Debug.xcconfig with your Development Team ID: $$TEAM_ID"; \
	fi


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


list-devices:
	fastlane ios list_devices


# Usage:
#	make benchmark-devices										# Benchmark all connected devices
#	make benchmark-devices DEBUG=true							# Benchmark all connected devices with small test matrix
#	make benchmark-devices DEVICES="iPhone 15 Pro Max,My Mac"	# Benchmark specific device names from `make list-devices`
DEVICES ?=
DEBUG ?= false
benchmark-devices: generate-whisperax-xcconfig
	@if [ -n "$(DEVICES)" ]; then \
		echo "Benchmarking specific devices: $(DEVICES)"; \
		fastlane benchmark devices:"$(DEVICES)" debug:$(DEBUG); \
	else \
		echo "Benchmarking all connected devices"; \
		fastlane benchmark debug:$(DEBUG); \
	fi

upload-benchmark-results:
	@echo "Uploading benchmark results..."
	@fastlane upload_results

clean-package-caches:
	@trash ~/Library/Developer/Xcode/DerivedData/WhisperKit* || true
	@swift package purge-cache
	@swift package reset