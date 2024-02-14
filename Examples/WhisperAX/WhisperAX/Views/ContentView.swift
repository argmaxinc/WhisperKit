//
//  ContentView.swift
//  WhisperAX
//
//  Created by Zach Nagengast on 1/16/24.
//

import SwiftUI
import WhisperKit
#if canImport(UIKit)
    import UIKit
#elseif canImport(AppKit)
    import AppKit
#endif
import AVFoundation

struct ContentView: View {
    @State var whisperKit: WhisperKit? = nil
    @State var isRecording: Bool = false
    @State var isTranscribing: Bool = false
    @State var currentText: String = ""
    // TODO: Make this configurable in the UI
    @State var modelStorage: String = "huggingface/models/argmaxinc/whisperkit-coreml"

    @State private var modelState: ModelState = .unloaded
    @State private var localModels: [String] = []
    @State private var localModelPath: String = ""
    @State private var availableModels: [String] = []
    @State private var availableLanguages: [String] = []
    @State private var disabledModels: [String] = WhisperKit.recommendedModels().disabled

    @AppStorage("selectedModel") private var selectedModel: String = WhisperKit.recommendedModels().default
    @AppStorage("selectedTab") private var selectedTab: String = "Transcribe"
    @AppStorage("selectedTask") private var selectedTask: String = "transcribe"
    @AppStorage("selectedLanguage") private var selectedLanguage: String = "english"
    @AppStorage("repoName") private var repoName: String = "argmaxinc/whisperkit-coreml"
    @AppStorage("enableTimestamps") private var enableTimestamps: Bool = true
    @AppStorage("enablePromptPrefill") private var enablePromptPrefill: Bool = true
    @AppStorage("enableCachePrefill") private var enableCachePrefill: Bool = true
    @AppStorage("enableSpecialCharacters") private var enableSpecialCharacters: Bool = false
    @AppStorage("enableEagerDecoder") private var enableEagerDecoder: Bool = false
    @AppStorage("temperatureStart") private var temperatureStart: Double = 0
    @AppStorage("fallbackCount") private var fallbackCount: Double = 4
    @AppStorage("compressionCheckWindow") private var compressionCheckWindow: Double = 20
    @AppStorage("sampleLength") private var sampleLength: Double = 224
    @AppStorage("silenceThreshold") private var silenceThreshold: Double = 0.3
    @AppStorage("useVAD") private var useVAD: Bool = true

    @State private var loadingProgressValue: Float = 0.0
    @State private var specializationProgressRatio: Float = 0.7
    @State private var isFilePickerPresented = false
    @State private var firstTokenTime: TimeInterval = 0
    @State private var pipelineStart: TimeInterval = 0
    @State private var realTimeFactor: TimeInterval = 0
    @State private var tokensPerSecond: TimeInterval = 0
    @State private var currentLag: TimeInterval = 0
    @State private var currentFallbacks: Int = 0
    @State private var lastBufferSize: Int = 0
    @State private var lastConfirmedSegmentEndSeconds: Float = 0
    @State private var requiredSegmentsForConfirmation: Int = 2
    @State private var bufferEnergy: [Float] = []
    @State private var confirmedSegments: [TranscriptionSegment] = []
    @State private var unconfirmedSegments: [TranscriptionSegment] = []
    @State private var unconfirmedText: [String] = []

    @State private var showAdvancedOptions: Bool = false
    @State private var transcriptionTask: Task<Void, Never>? = nil

    @State private var selectedCategoryId: MenuItem.ID?
    private var menu = [
        MenuItem(name: "Transcribe", image: "book.pages"),
        MenuItem(name: "Stream", image: "waveform.badge.mic"),
    ]

    struct MenuItem: Identifiable, Hashable {
        var id = UUID()
        var name: String
        var image: String
    }

    // MARK: Views

    func resetState() {
        isRecording = false
        isTranscribing = false
        whisperKit?.audioProcessor.stopRecording()
        currentText = ""
        unconfirmedText = []

        firstTokenTime = 0
        pipelineStart = 0
        realTimeFactor = 0
        tokensPerSecond = 0
        currentLag = 0
        currentFallbacks = 0
        lastBufferSize = 0
        lastConfirmedSegmentEndSeconds = 0
        requiredSegmentsForConfirmation = 2
        bufferEnergy = []
        confirmedSegments = []
        unconfirmedSegments = []
    }

    var body: some View {
        NavigationSplitView(columnVisibility: Binding.constant(.all)) {
            modelSelectorView
                .padding()
            Spacer()
            List(menu, selection: $selectedCategoryId) { item in
                HStack {
                    Image(systemName: item.image)
                    Text(item.name)
                        .font(.system(.title3))
                        .bold()
                }
            }
            .disabled(modelState != .loaded)
            .foregroundColor(modelState != .loaded ? .secondary : .primary)
            .navigationTitle("WhisperAX")
            .navigationSplitViewColumnWidth(min: 300, ideal: 350)
        } detail: {
            VStack {
                #if os(iOS)
                    modelSelectorView
                        .padding()
                    transcriptionView
                #elseif os(macOS)
                    VStack(alignment: .leading) {
                        transcriptionView
                    }
                    .padding()
                #endif
                controlsView
            }
            .toolbar(content: {
                ToolbarItem {
                    Button {
                        let fullTranscript = formatSegments(confirmedSegments + unconfirmedSegments, withTimestamps: enableTimestamps).joined(separator: "\n")
                        #if os(iOS)
                            UIPasteboard.general.string = fullTranscript
                        #elseif os(macOS)
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(fullTranscript, forType: .string)
                        #endif
                    } label: {
                        Label("Copy Text", systemImage: "doc.on.doc")
                    }
                    .foregroundColor(.primary)
                    .frame(minWidth: 0, maxWidth: .infinity)
                }
            })
        }
        .onAppear {
            Task {
                whisperKit = try await WhisperKit(verbose: true, logLevel: .debug)
            }

            #if os(macOS)
                selectedCategoryId = menu.first(where: { $0.name == selectedTab })?.id
            #endif
            fetchModels()
        }
    }

    // MARK: - Transcription

    var transcriptionView: some View {
        VStack {
            ScrollView(.horizontal) {
                HStack(spacing: 1) {
                    let startIndex = max(bufferEnergy.count - 300, 0)
                    ForEach(Array(bufferEnergy.enumerated())[startIndex...], id: \.element) { _, energy in
                        ZStack {
                            RoundedRectangle(cornerRadius: 2)
                                .frame(width: 2, height: CGFloat(energy) * 24)
                        }
                        .frame(maxHeight: 24)
                        .background(energy > Float(silenceThreshold) ? Color.green.opacity(0.2) : Color.red.opacity(0.2))
                    }
                }
            }
            .defaultScrollAnchor(.trailing)
            .frame(height: 24)
            .scrollIndicators(.never)
            ScrollView {
                VStack(alignment: .leading) {
                    ForEach(Array(confirmedSegments.enumerated()), id: \.element) { _, segment in
                        let timestampText = enableTimestamps ? "[\(String(format: "%.2f", segment.start)) --> \(String(format: "%.2f", segment.end))]" : ""
                        Text(timestampText + segment.text)
                            .font(.headline)
                            .fontWeight(.bold)
                            .tint(.green)
                            .multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    ForEach(Array(unconfirmedSegments.enumerated()), id: \.element) { _, segment in
                        let timestampText = enableTimestamps ? "[\(String(format: "%.2f", segment.start)) --> \(String(format: "%.2f", segment.end))]" : ""
                        Text(timestampText + segment.text)
                            .font(.headline)
                            .fontWeight(.light)
                            .multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    Text("\(unconfirmedText.joined(separator: "\n"))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.leading)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Text("\(currentText)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.leading)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .frame(maxWidth: .infinity)
            .defaultScrollAnchor(.bottom)
            .padding()
        }
    }

    // MARK: - Models

    var modelSelectorView: some View {
        Group {
            VStack {
                HStack {
                    Image(systemName: "circle.fill")
                        .foregroundStyle(modelState == .loaded ? .green : (modelState == .unloaded ? .red : .yellow))
                        .symbolEffect(.variableColor, isActive: modelState != .loaded && modelState != .unloaded)
                    Text(modelState.description)

                    Spacer()

                    if availableModels.count > 0 {
                        Picker("", selection: $selectedModel) {
                            ForEach(availableModels, id: \.self) { model in
                                HStack {
                                    let modelIcon = localModels.contains { $0 == model.description } ? "checkmark.circle" : "arrow.down.circle.dotted"
                                    Text("\(Image(systemName: modelIcon)) \(model.description)").tag(model.description)
                                }
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .onChange(of: selectedModel, initial: false) { _, _ in
                            modelState = .unloaded
                        }
                    } else {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle())
                            .scaleEffect(0.5)
                    }

                    #if os(macOS)
                        Button(action: {
                            if let folder = whisperKit?.modelFolder {
                                NSWorkspace.shared.open(folder)
                            }
                        }, label: {
                            Image(systemName: "folder")
                        })
                        .buttonStyle(BorderlessButtonStyle())
                    #endif
                    Button(action: {
                        if let url = URL(string: "https://huggingface.co/\(repoName)") {
                            #if os(macOS)
                                NSWorkspace.shared.open(url)
                            #else
                                UIApplication.shared.open(url)
                            #endif
                        }
                    }, label: {
                        Image(systemName: "link.circle")
                    })
                    .buttonStyle(BorderlessButtonStyle())
                }

                if modelState == .unloaded {
                    Divider()
                    Button {
                        resetState()
                        loadModel(selectedModel)
                        modelState = .loading
                    } label: {
                        Text("Load Model")
                            .frame(maxWidth: .infinity)
                            .frame(height: 40)
                    }
                    .buttonStyle(.borderedProminent)
                } else if loadingProgressValue < 1.0 {
                    VStack {
                        HStack {
                            ProgressView(value: loadingProgressValue, total: 1.0)
                                .progressViewStyle(LinearProgressViewStyle())
                                .frame(maxWidth: .infinity)

                            Text(String(format: "%.1f%%", loadingProgressValue * 100))
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                        if modelState == .prewarming {
                            Text("Specializing \(selectedModel) for your device...\nThis can take several minutes on first load")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Controls

    var controlsView: some View {
        VStack {
            basicSettingsView

            if let selectedCategoryId, let item = menu.first(where: { $0.id == selectedCategoryId }) {
                switch item.name {
                case "Transcribe":
                    VStack {
                        HStack {
                            Button {
                                resetState()
                            } label: {
                                Label("Reset", systemImage: "arrow.clockwise")
                            }
                            .buttonStyle(.borderless)
                            Spacer()
                            Button {
                                showAdvancedOptions.toggle()
                            } label: {
                                Label("Settings", systemImage: "slider.horizontal.3")
                            }
                            .buttonStyle(.borderless)
                        }

                        HStack {
                            let color: Color = modelState != .loaded ? .gray : .red
                            Button(action: {
                                withAnimation {
                                    selectFile()
                                }
                            }) {
                                Text("FROM FILE")
                                    .font(.headline)
                                    .foregroundColor(color)
                                    .padding()
                                    .cornerRadius(40)
                                    .frame(minWidth: 70, minHeight: 70)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 40)
                                            .stroke(color, lineWidth: 4)
                                    )
                            }
                            .fileImporter(
                                isPresented: $isFilePickerPresented,
                                allowedContentTypes: [.audio],
                                allowsMultipleSelection: false,
                                onCompletion: handleFilePicker
                            )
                            .lineLimit(1)
                            .contentTransition(.symbolEffect(.replace))
                            .buttonStyle(BorderlessButtonStyle())
                            .disabled(modelState != .loaded)
                            .frame(minWidth: 0, maxWidth: .infinity)
                            .padding()

                            Button(action: {
                                withAnimation {
                                    toggleRecording(shouldLoop: false)
                                }
                            }) {
                                if !isRecording {
                                    Text("RECORD")
                                        .font(.headline)
                                        .foregroundColor(color)
                                        .padding()
                                        .cornerRadius(40)
                                        .frame(minWidth: 70, minHeight: 70)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 40)
                                                .stroke(color, lineWidth: 4)
                                        )
                                } else {
                                    Image(systemName: "stop.circle.fill")
                                        .resizable()
                                        .scaledToFit()
                                        .frame(width: 70, height: 70)
                                        .padding()
                                        .foregroundColor(modelState != .loaded ? .gray : .red)
                                }
                            }
                            .lineLimit(1)
                            .contentTransition(.symbolEffect(.replace))
                            .buttonStyle(BorderlessButtonStyle())
                            .disabled(modelState != .loaded)
                            .frame(minWidth: 0, maxWidth: .infinity)
                            .padding()
                        }
                    }
                case "Stream":
                    HStack {
                        Button {
                            resetState()
                        } label: {
                            Label("Reset", systemImage: "arrow.clockwise")
                        }
                        .frame(minWidth: 0, maxWidth: .infinity)
                        .buttonStyle(.borderless)

                        Button {
                            withAnimation {
                                toggleRecording(shouldLoop: true)
                            }
                        } label: {
                            Image(systemName: !isRecording ? "record.circle" : "stop.circle.fill")
                                .resizable()
                                .scaledToFit()
                                .frame(width: 70, height: 70)
                                .padding()
                                .foregroundColor(modelState != .loaded ? .gray : .red)
                        }
                        .contentTransition(.symbolEffect(.replace))
                        .buttonStyle(BorderlessButtonStyle())
                        .disabled(modelState != .loaded)
                        .frame(minWidth: 0, maxWidth: .infinity)

                        Button {
                            showAdvancedOptions.toggle()
                        } label: {
                            Label("Settings", systemImage: "slider.horizontal.3")
                        }
                        .frame(minWidth: 0, maxWidth: .infinity)
                        .buttonStyle(.borderless)
                    }
                default:
                    EmptyView()
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal)
        .sheet(isPresented: $showAdvancedOptions, content: {
            advancedSettingsView
                .presentationDetents([.medium, .large])
                .presentationBackgroundInteraction(.enabled)
                .presentationContentInteraction(.scrolls)
        })
    }

    var basicSettingsView: some View {
        VStack {
            HStack {
                Picker("", selection: $selectedTask) {
                    ForEach(DecodingTask.allCases, id: \.self) { task in
                        Text(task.description.capitalized).tag(task.description)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .disabled(!(whisperKit?.modelVariant.isMultilingual ?? false))
            }
            .padding(.horizontal)

            LabeledContent {
                Picker("", selection: $selectedLanguage) {
                    ForEach(availableLanguages, id: \.self) { language in
                        Text(language.description).tag(language.description)
                    }
                }
                .disabled(!(whisperKit?.modelVariant.isMultilingual ?? false))
            } label: {
                Label("Language", systemImage: "globe")
            }
            .padding(.horizontal)
            .padding(.top)

            HStack {
                Text(realTimeFactor.formatted(.number.precision(.fractionLength(3))) + " RTF")
                    .font(.system(.body))
                    .lineLimit(1)
                Spacer()
                Text(tokensPerSecond.formatted(.number.precision(.fractionLength(0))) + " tok/s")
                    .font(.system(.body))
                    .lineLimit(1)
                Spacer()
                Text("First token: " + (firstTokenTime - pipelineStart).formatted(.number.precision(.fractionLength(2))) + "s")
                    .font(.system(.body))
                    .lineLimit(1)
            }
            .padding()
            .frame(maxWidth: .infinity)
        }
    }

    var advancedSettingsView: some View {
        #if os(iOS)
            NavigationView {
                settingsForm
                    .navigationBarTitleDisplayMode(.inline)
            }
        #else
            VStack {
                Text("Decoding Options")
                    .font(.title2)
                    .padding()
                settingsForm
                    .frame(minWidth: 500, minHeight: 500)
            }
        #endif
    }

    var settingsForm: some View {
        List {
            HStack {
                Text("Show Timestamps")
                InfoButton("Toggling this will include/exclude timestamps in both the UI and the prefill tokens.\nEither <|notimestamps|> or <|0.00|> will be forced based on this setting unless \"Prompt Prefill\" is de-selected.")
                Spacer()
                Toggle("", isOn: $enableTimestamps)
            }
            .padding(.horizontal)

            HStack {
                Text("Special Characters")
                InfoButton("Toggling this will include/exclude special characters in the transcription text.")
                Spacer()
                Toggle("", isOn: $enableSpecialCharacters)
            }
            .padding(.horizontal)

            HStack {
                Text("Prompt Prefill")
                InfoButton("When Prompt Prefill is on, it will force the task, language, and timestamp tokens in the decoding loop. \nToggle it off if you'd like the model to generate those tokens itself instead.")
                Spacer()
                Toggle("", isOn: $enablePromptPrefill)
            }
            .padding(.horizontal)

            HStack {
                Text("Cache Prefill")
                InfoButton("When Cache Prefill is on, the decoder will try to use a lookup table of pre-computed KV caches instead of computing them during the decoding loop. \nThis allows the model to skip the compute required to force the initial prefill tokens, and can speed up inference")
                Spacer()
                Toggle("", isOn: $enableCachePrefill)
            }
            .padding(.horizontal)

            VStack {
                Text("Starting Temperature:")
                HStack {
                    Slider(value: $temperatureStart, in: 0...1, step: 0.1)
                    Text(temperatureStart.formatted(.number))
                    InfoButton("Controls the initial randomness of the decoding loop token selection.\nA higher temperature will result in more random choices for tokens, and can improve accuracy.")
                }
            }
            .padding(.horizontal)

            VStack {
                Text("Max Fallback Count:")
                HStack {
                    Slider(value: $fallbackCount, in: 0...5, step: 1)
                    Text(fallbackCount.formatted(.number))
                        .frame(width: 30)
                    InfoButton("Controls how many times the decoder will fallback to a higher temperature if any of the decoding thresholds are exceeded.\n Higher values will cause the decoder to run multiple times on the same audio, which can improve accuracy at the cost of speed.")
                }
            }
            .padding(.horizontal)

            VStack {
                Text("Compression Check Tokens")
                HStack {
                    Slider(value: $compressionCheckWindow, in: 0...100, step: 5)
                    Text(compressionCheckWindow.formatted(.number))
                        .frame(width: 30)
                    InfoButton("Amount of tokens to use when checking for whether the model is stuck in a repetition loop.\nRepetition is checked by using zlib compressed size of the text compared to non-compressed value.\n Lower values will catch repetitions sooner, but too low will miss repetition loops of phrases longer than the window.")
                }
            }
            .padding(.horizontal)

            VStack {
                Text("Max Tokens Per Loop")
                HStack {
                    Slider(value: $sampleLength, in: 0...Double(min(whisperKit?.textDecoder.kvCacheMaxSequenceLength ?? WhisperKit.maxTokenContext, WhisperKit.maxTokenContext)), step: 10)
                    Text(sampleLength.formatted(.number))
                        .frame(width: 30)
                    InfoButton("Maximum number of tokens to generate per loop.\nCan be lowered based on the type of speech in order to further prevent repetition loops from going too long.")
                }
            }
            .padding(.horizontal)

            VStack {
                Text("Silence Threshold")
                HStack {
                    Slider(value: $silenceThreshold, in: 0...1, step: 0.05)
                    Text(silenceThreshold.formatted(.number))
                        .frame(width: 30)
                    InfoButton("Relative silence threshold for the audio. \n Baseline is set by the quietest 100ms in the previous 2 seconds.")
                }
            }
            .padding(.horizontal)
        }
        .navigationTitle("Decoding Options")
        .toolbar(content: {
            ToolbarItem {
                Button {
                    showAdvancedOptions = false
                } label: {
                    Label("Done", systemImage: "xmark.circle.fill")
                        .foregroundColor(.primary)
                }
            }
        })
    }

    struct InfoButton: View {
        var infoText: String
        @State private var showInfo = false

        init(_ infoText: String) {
            self.infoText = infoText
        }

        var body: some View {
            Button(action: {
                self.showInfo = true
            }) {
                Image(systemName: "info.circle")
                    .foregroundColor(.blue)
            }
            .popover(isPresented: $showInfo) {
                Text(infoText)
                    .padding()
            }
            .buttonStyle(BorderlessButtonStyle())
        }
    }

    // MARK: Logic

    func fetchModels() {
        availableModels = [selectedModel]

        // First check what's already downloaded
        if let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let modelPath = documents.appendingPathComponent(modelStorage).path

            // Check if the directory exists
            if FileManager.default.fileExists(atPath: modelPath) {
                localModelPath = modelPath
                do {
                    let downloadedModels = try FileManager.default.contentsOfDirectory(atPath: modelPath)
                    for model in downloadedModels where !localModels.contains(model) && model.starts(with: "openai") {
                        localModels.append(model)
                    }
                } catch {
                    print("Error enumerating files at \(modelPath): \(error.localizedDescription)")
                }
            }
        }

        localModels = WhisperKit.formatModelFiles(localModels)
        for model in localModels {
            if !availableModels.contains(model),
               !disabledModels.contains(model)
            {
                availableModels.append(model)
            }
        }

        print("Found locally: \(localModels)")
        print("Previously selected model: \(selectedModel)")

        Task {
            let remoteModels = try await WhisperKit.fetchAvailableModels(from: repoName)
            for model in remoteModels {
                if !availableModels.contains(model),
                   !disabledModels.contains(model)
                {
                    availableModels.append(model)
                }
            }
        }
    }

    func requestMicrophoneIfNeeded() async -> Bool {
        let microphoneStatus = AVCaptureDevice.authorizationStatus(for: .audio)

        switch microphoneStatus {
        case .notDetermined:
            return await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    continuation.resume(returning: granted)
                }
            }
        case .restricted, .denied:
            print("Microphone access denied")
            return false
        case .authorized:
            return true
        @unknown default:
            fatalError("Unknown authorization status")
        }
    }

    func loadModel(_ model: String, redownload: Bool = false) {
        print("Selected Model: \(UserDefaults.standard.string(forKey: "selectedModel") ?? "nil")")

        whisperKit = nil
        Task {
            whisperKit = try await WhisperKit(
                verbose: true,
                logLevel: .debug,
                prewarm: false,
                load: false,
                download: false
            )
            guard let whisperKit = whisperKit else {
                return
            }

            var folder: URL?

            // Check if the model is available locally
            if localModels.contains(model) && !redownload {
                // Get local model folder URL from localModels
                // TODO: Make this configurable in the UI
                folder = URL(fileURLWithPath: localModelPath).appendingPathComponent("openai_whisper-\(model)")
            } else {
                // Download the model
                folder = try await WhisperKit.download(variant: model, from: repoName, progressCallback: { progress in
                    DispatchQueue.main.async {
                        loadingProgressValue = Float(progress.fractionCompleted) * specializationProgressRatio
                        modelState = .downloading
                    }
                })
            }

            if let modelFolder = folder {
                whisperKit.modelFolder = modelFolder

                await MainActor.run {
                    // Set the loading progress to 90% of the way after prewarm
                    loadingProgressValue = specializationProgressRatio
                    modelState = .prewarming
                }

                let progressBarTask = Task {
                    await updateProgressBar(targetProgress: 0.9, maxTime: 240)
                }

                // Prewarm models
                do {
                    try await whisperKit.prewarmModels()
                    progressBarTask.cancel()
                } catch {
                    print("Error prewarming models, retrying: \(error.localizedDescription)")
                    progressBarTask.cancel()
                    if !redownload {
                        loadModel(model, redownload: true)
                        return
                    } else {
                        // Redownloading failed, error out
                        modelState = .unloaded
                        return
                    }
                }

                await MainActor.run {
                    // Set the loading progress to 90% of the way after prewarm
                    loadingProgressValue = specializationProgressRatio + 0.9 * (1 - specializationProgressRatio)
                    modelState = .loading
                }

                try await whisperKit.loadModels()

                await MainActor.run {
                    availableLanguages = whisperKit.tokenizer?.langauges.map { $0.key }.sorted() ?? ["english"]
                    loadingProgressValue = 1.0
                    modelState = whisperKit.modelState
                }
            }
        }
    }

    func updateProgressBar(targetProgress: Float, maxTime: TimeInterval) async {
        let initialProgress = loadingProgressValue
        let decayConstant = -log(1 - targetProgress) / Float(maxTime)

        let startTime = Date()

        while true {
            let elapsedTime = Date().timeIntervalSince(startTime)

            // Break down the calculation
            let decayFactor = exp(-decayConstant * Float(elapsedTime))
            let progressIncrement = (1 - initialProgress) * (1 - decayFactor)
            let currentProgress = initialProgress + progressIncrement

            await MainActor.run {
                loadingProgressValue = currentProgress
            }

            if currentProgress >= targetProgress {
                break
            }

            do {
                try await Task.sleep(nanoseconds: 100_000_000)
            } catch {
                break
            }
        }
    }

    func selectFile() {
        isFilePickerPresented = true
    }

    func handleFilePicker(result: Result<[URL], Error>) {
        switch result {
        case let .success(urls):
            guard let selectedFileURL = urls.first else { return }
            if selectedFileURL.startAccessingSecurityScopedResource() {
                do {
                    // Access the document data from the file URL
                    let audioFileData = try Data(contentsOf: selectedFileURL)

                    // Create a unique file name to avoid overwriting any existing files
                    let uniqueFileName = UUID().uuidString + "." + selectedFileURL.pathExtension

                    // Construct the temporary file URL in the app's temp directory
                    let tempDirectoryURL = FileManager.default.temporaryDirectory
                    let localFileURL = tempDirectoryURL.appendingPathComponent(uniqueFileName)

                    // Write the data to the temp directory
                    try audioFileData.write(to: localFileURL)

                    print("File saved to temporary directory: \(localFileURL)")

                    transcribeFile(path: selectedFileURL.path)
                } catch {
                    print("File selection error: \(error.localizedDescription)")
                }
            }
        case let .failure(error):
            print("File selection error: \(error.localizedDescription)")
        }
    }

    func transcribeFile(path: String) {
        resetState()
        whisperKit?.audioProcessor = AudioProcessor()
        Task {
            do {
                try await transcribeCurrentFile(path: path)
            } catch {
                print("File selection error: \(error.localizedDescription)")
            }
        }
    }

    func toggleRecording(shouldLoop: Bool) {
        isRecording.toggle()

        if isRecording {
            resetState()
            startRecording(shouldLoop)
        } else {
            stopRecording(shouldLoop)
        }
    }

    func startRecording(_ loop: Bool) {
        if let audioProcessor = whisperKit?.audioProcessor {
            Task(priority: .userInitiated) {
                guard await requestMicrophoneIfNeeded() else {
                    print("Microphone access was not granted.")
                    return
                }

                try? audioProcessor.startRecordingLive { _ in
                    DispatchQueue.main.async {
                        bufferEnergy = whisperKit?.audioProcessor.relativeEnergy ?? []
                    }
                }

                // Delay the timer start by 1 second
                isRecording = true
                isTranscribing = true
                if loop {
                    realtimeLoop()
                }
            }
        }
    }

    func stopRecording(_ loop: Bool) {
        isRecording = false
        stopRealtimeTranscription()
        if let audioProcessor = whisperKit?.audioProcessor {
            audioProcessor.stopRecording()
        }

        // If not looping, transcribe the full buffer
        if !loop {
            Task {
                do {
                    try await transcribeCurrentBuffer()
                } catch {
                    print("Error: \(error.localizedDescription)")
                }
            }
        }
    }

    // MARK: Transcribe Logic

    func transcribeCurrentFile(path: String) async throws {
        guard let audioFileBuffer = AudioProcessor.loadAudio(fromPath: path) else {
            return
        }

        let audioFileSamples = AudioProcessor.convertBufferToArray(buffer: audioFileBuffer)
        let transcription = try await transcribeAudioSamples(audioFileSamples)

        await MainActor.run {
            currentText = ""
            unconfirmedText = []
            guard let segments = transcription?.segments else {
                return
            }

            self.tokensPerSecond = transcription?.timings?.tokensPerSecond ?? 0
            self.realTimeFactor = transcription?.timings?.realTimeFactor ?? 0
            self.firstTokenTime = transcription?.timings?.firstTokenTime ?? 0
            self.pipelineStart = transcription?.timings?.pipelineStart ?? 0
            self.currentLag = transcription?.timings?.decodingLoop ?? 0

            self.confirmedSegments = segments
        }
    }

    func transcribeAudioSamples(_ samples: [Float]) async throws -> TranscriptionResult? {
        guard let whisperKit = whisperKit else { return nil }

        let languageCode = whisperKit.tokenizer?.langauges[selectedLanguage] ?? "en"
        let task: DecodingTask = selectedTask == "transcribe" ? .transcribe : .translate
        let seekClip = [lastConfirmedSegmentEndSeconds]

        let options = DecodingOptions(
            verbose: false,
            task: task,
            language: languageCode,
            temperatureFallbackCount: 3, // limit fallbacks for realtime
            sampleLength: Int(sampleLength), // reduced sample length for realtime
            usePrefillPrompt: enablePromptPrefill,
            usePrefillCache: enableCachePrefill,
            skipSpecialTokens: !enableSpecialCharacters,
            withoutTimestamps: !enableTimestamps,
            clipTimestamps: seekClip
        )

        // Early stopping checks
        let decodingCallback: ((TranscriptionProgress) -> Bool?) = { progress in
            DispatchQueue.main.async {
                let fallbacks = Int(progress.timings.totalDecodingFallbacks)
                if progress.text.count < currentText.count {
                    if fallbacks == self.currentFallbacks {
                        self.unconfirmedText.append(currentText)
                    } else {
                        print("Fallback occured: \(fallbacks)")
                    }
                }
                self.currentText = progress.text
                self.currentFallbacks = fallbacks
            }
            // Check early stopping
            let currentTokens = progress.tokens
            let checkWindow = Int(compressionCheckWindow)
            if currentTokens.count > checkWindow {
                let checkTokens: [Int] = currentTokens.suffix(checkWindow)
                let compressionRatio = compressionRatio(of: checkTokens)
                if compressionRatio > options.compressionRatioThreshold! {
                    return false
                }
            }
            if progress.avgLogprob! < options.logProbThreshold! {
                return false
            }

            return nil
        }

        let transcription = try await whisperKit.transcribe(audioArray: samples, decodeOptions: options, callback: decodingCallback)
        return transcription
    }

    // MARK: Streaming Logic

    func realtimeLoop() {
        transcriptionTask = Task {
            while isRecording && isTranscribing {
                do {
                    try await transcribeCurrentBuffer()
                } catch {
                    print("Error: \(error.localizedDescription)")
                    break
                }
            }
        }
    }

    func stopRealtimeTranscription() {
        isTranscribing = false
        transcriptionTask?.cancel()
    }

    func transcribeCurrentBuffer() async throws {
        guard let whisperKit = whisperKit else { return }

        // Retrieve the current audio buffer from the audio processor
        let currentBuffer = whisperKit.audioProcessor.audioSamples

        // Calculate the size and duration of the next buffer segment
        let nextBufferSize = currentBuffer.count - lastBufferSize
        let nextBufferSeconds = Float(nextBufferSize) / Float(WhisperKit.sampleRate)

        // Only run the transcribe if the next buffer has at least 1 second of audio
        guard nextBufferSeconds > 1 else {
            await MainActor.run {
                if currentText == "" {
                    currentText = "Waiting for speech..."
                }
            }
            try await Task.sleep(nanoseconds: 100_000_000) // sleep for 100ms for next buffer
            return
        }

        if useVAD {
            // Retrieve the current relative energy values from the audio processor
            let currentRelativeEnergy = whisperKit.audioProcessor.relativeEnergy

            // Calculate the number of energy values to consider based on the duration of the next buffer
            // Each energy value corresponds to 1 buffer length (100ms of audio), hence we divide by 0.1
            let energyValuesToConsider = Int(nextBufferSeconds / 0.1)

            // Extract the relevant portion of energy values from the currentRelativeEnergy array
            let nextBufferEnergies = currentRelativeEnergy.suffix(energyValuesToConsider)

            // Determine the number of energy values to check for voice presence
            // Considering up to the last 1 second of audio, which translates to 10 energy values
            let numberOfValuesToCheck = max(10, nextBufferEnergies.count - 10)

            // Check if any of the energy values in the considered range exceed the silence threshold
            // This indicates the presence of voice in the buffer
            let voiceDetected = nextBufferEnergies.prefix(numberOfValuesToCheck).contains { $0 > Float(silenceThreshold) }

            // Only run the transcribe if the next buffer has voice
            guard voiceDetected else {
                await MainActor.run {
                    if currentText == "" {
                        currentText = "Waiting for speech..."
                    }
                }

//                if nextBufferSeconds > 30 {
//                    // This is a completely silent segment of 30s, so we can purge the audio and confirm anything pending
//                    lastConfirmedSegmentEndSeconds = 0
//                    whisperKit.audioProcessor.purgeAudioSamples(keepingLast: 2 * WhisperKit.sampleRate) // keep last 2s to include VAD overlap
//                    currentBuffer = whisperKit.audioProcessor.audioSamples
//                    lastBufferSize = 0
//                    confirmedSegments.append(contentsOf: unconfirmedSegments)
//                    unconfirmedSegments = []
//                }

                // Sleep for 100ms and check the next buffer
                try await Task.sleep(nanoseconds: 100_000_000)
                return
            }
        }

        // Run transcribe
        lastBufferSize = currentBuffer.count

        let transcription = try await transcribeAudioSamples(Array(currentBuffer))

        // We need to run this next part on the main thread
        await MainActor.run {
            currentText = ""
            unconfirmedText = []
            guard let segments = transcription?.segments else {
                return
            }

            self.tokensPerSecond = transcription?.timings?.tokensPerSecond ?? 0
            self.realTimeFactor = transcription?.timings?.realTimeFactor ?? 0
            self.firstTokenTime = transcription?.timings?.firstTokenTime ?? 0
            self.pipelineStart = transcription?.timings?.pipelineStart ?? 0
            self.currentLag = transcription?.timings?.decodingLoop ?? 0

            // Logic for moving segments to confirmedSegments
            if segments.count > requiredSegmentsForConfirmation {
                // Calculate the number of segments to confirm
                let numberOfSegmentsToConfirm = segments.count - requiredSegmentsForConfirmation

                // Confirm the required number of segments
                let confirmedSegmentsArray = Array(segments.prefix(numberOfSegmentsToConfirm))
                let remainingSegments = Array(segments.suffix(requiredSegmentsForConfirmation))

                // Update lastConfirmedSegmentEnd based on the last confirmed segment
                if let lastConfirmedSegment = confirmedSegmentsArray.last, lastConfirmedSegment.end > lastConfirmedSegmentEndSeconds {
                    lastConfirmedSegmentEndSeconds = lastConfirmedSegment.end

                    // Add confirmed segments to the confirmedSegments array
                    if !self.confirmedSegments.contains(confirmedSegmentsArray) {
                        self.confirmedSegments.append(contentsOf: confirmedSegmentsArray)
                    }
                }

                // Update transcriptions to reflect the remaining segments
                self.unconfirmedSegments = remainingSegments
            } else {
                // Handle the case where segments are fewer or equal to required
                self.unconfirmedSegments = segments
            }
        }
    }
}

#Preview {
    ContentView()
    #if os(macOS)
        .frame(width: 800, height: 500)
    #endif
}
