include_directories(${DYNAMATIC_SOURCE_DIR}/experimental/include/experimental/Transforms/EE)

add_dynamatic_library(DynamaticEE
  HandshakeEE.cpp
  PostEE.cpp
  PreEE.cpp
  JSONImporter.cpp
  EEPostBuffering.cpp
  EECutCondDep.cpp
  EEGamma.cpp
  EEGammaPostBuffering.cpp
  PreEEGamma.cpp
  EELib.cpp
  SpecAdaptor.cpp
  SpecPostAdaptor.cpp

  DEPENDS
  DynamaticExperimentalTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
  DynamaticHandshake
  DynamaticSupport
  DynamaticExperimentalSupport
  MaterializationUtil
  )

