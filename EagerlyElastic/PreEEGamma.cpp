#include "PreEEGamma.h"
#include "EELib.h"
#include "JSONImporter.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm::sys;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::handshake;
using namespace dynamatic::experimental;
using namespace dynamatic::experimental::EE;

namespace dynamatic {
namespace experimental {
namespace EE {

// Implement the base class and auto-generated create functions.
// Must be called from the .cpp file to avoid multiple definitions
#define GEN_PASS_DEF_PREEEGAMMA
#include "experimental/Transforms/Passes.h.inc"

} // namespace EE
} // namespace experimental
} // namespace dynamatic

struct PreEEGammaPass
    : public dynamatic::experimental::EE::impl::PreEEGammaBase<PreEEGammaPass> {
  using PreEEGammaBase<PreEEGammaPass>::PreEEGammaBase;
  void runDynamaticPass() override;
};

void PreEEGammaPass::runDynamaticPass() {
  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  introduceGSAMux(funcOp, branchBB);

  OpBuilder builder(funcOp->getContext());
  for (auto mux : funcOp.getOps<MuxOp>()) {
    if (getLogicBB(mux) != mergeBB)
      continue;

    mux->setAttr("EE_loop_cond_mux", builder.getBoolAttr(true));
    mux->setAttr("EE_gsa_mux", builder.getUnitAttr());
  }

  if (failed(replaceBranchesWithPassers(funcOp, branchBB))) {
    funcOp.emitError("Failed to replace branches in BB 1 with passers");
    return signalPassFailure();
  }
}
