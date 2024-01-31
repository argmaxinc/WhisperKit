.PHONY: setup download-models build build-cli test 

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
	@$(PIP_COMMAND) install -U huggingface_hub
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

download-models:
	@echo "Downloading compressed models..."
	@mkdir -p $(BASE_COMPILED_DIR)
	@if [ -d "$(MODEL_REPO_DIR)/.git" ]; then \
		echo "Repository exists, pulling latest changes..."; \
		cd $(MODEL_REPO_DIR) && git reset --hard origin/main; \
	else \
		echo "Repository not found, cloning..."; \
		$(eval HF_USERNAME := $(shell huggingface-cli whoami | head -n 1)) \
		git clone https://$(HF_USERNAME):$(HF_TOKEN)@hf.co/$(MODEL_REPO) $(MODEL_REPO_DIR); \
	fi


build:
	@echo "Building WhisperKit..."
	@swift build -v

build-cli:
	@echo "Building WhisperKit CLI..."
	@swift build -c release --product transcribe

test:
	@echo "Running tests..."
	@swift test -v
