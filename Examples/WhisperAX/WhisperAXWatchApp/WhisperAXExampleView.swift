//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import SwiftUI
import WhisperKit

struct WhisperAXWatchView: View {
    @State private var whisperKit: WhisperKit?
    @State private var transcription = "Tap below to start"

    var body: some View {
        VStack {
            Image(systemName: "mic.circle")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text(transcription)
                .lineLimit(nil)
                .multilineTextAlignment(.center)
                .padding()
                .fixedSize(horizontal: false, vertical: true)
            Button(whisperKit == nil ? "Loading WhisperKit..." : "Transcribe Example") {
                transcribeAudio()
            }
            .font(.footnote)
            .disabled(whisperKit == nil)
        }
        .padding()
        .onAppear {
            Task {
                try await prepareWhisper()
            }
        }
    }

    func prepareWhisper() async throws {
        let modelPath = Bundle.main.bundlePath
        whisperKit = try await WhisperKit(modelFolder: modelPath)
    }

    func transcribeAudio() {
        guard let whisperKit = whisperKit else { return }
        Task {
            // TODO: implement
        }
    }
}

#Preview {
    WhisperAXWatchView()
}
