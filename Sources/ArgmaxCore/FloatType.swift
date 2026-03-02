//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import CoreML

// MARK: - Platform-aware Float Type

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
public typealias FloatType = Float16
#else
public typealias FloatType = Float
#endif

#if (os(macOS) || targetEnvironment(macCatalyst)) && arch(arm64) && compiler(<6)
extension Float16: BNNSScalar {}
extension Float16: MLShapedArrayScalar {}
#endif
