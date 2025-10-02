#include "PreEE.h"
#include "EELib.h"
#include "JSONImporter.h"
#include "dynamatic/Dialect/Handshake/HandshakeInterfaces.h"
#include "dynamatic/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
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
#define GEN_PASS_DEF_PREEE
#include "experimental/Transforms/Passes.h.inc"

} // namespace EE
} // namespace experimental
} // namespace dynamatic

struct PreEEPass
    : public dynamatic::experimental::EE::impl::PreEEBase<PreEEPass> {
  using PreEEBase<PreEEPass>::PreEEBase;
  void runDynamaticPass() override;
};

void PreEEPass::runDynamaticPass() {
  // Parse json (jsonPath is a member variable handled by tablegen)
  auto bbOrFailure = readFromJSON(jsonPath);
  if (failed(bbOrFailure))
    return signalPassFailure();

  auto [loopBBs] = bbOrFailure.value();

  ModuleOp modOp = getOperation();

  // Support only one funcOp
  assert(std::distance(modOp.getOps<FuncOp>().begin(),
                       modOp.getOps<FuncOp>().end()) == 1 &&
         "Expected a single FuncOp in the module");

  FuncOp funcOp = *modOp.getOps<FuncOp>().begin();

  SmallVector<unsigned> exitingBBs;

  for (unsigned bb : loopBBs) {
    if (hasBranch(funcOp, bb) && !isExitingBBWithBranch(funcOp, bb, loopBBs)) {
      introduceGSAMux(funcOp, bb);
    }
  }

  // Replace branches with passers
  for (unsigned bb : loopBBs) {
    if (isExitingBBWithBranch(funcOp, bb, loopBBs))
      exitingBBs.push_back(bb);

    if (!hasBranch(funcOp, bb))
      continue;

    if (failed(replaceBranchesWithPassers(funcOp, bb)))
      return signalPassFailure();
  }

  Value loopCondition = calculateLoopCondition(funcOp, exitingBBs, loopBBs);

  loopCondition.dump();
  for (unsigned bb : exitingBBs) {
    llvm::errs() << "Exiting BB with branch: " << bb << "\n";
  }

  // Update the loop header (CMerge -> Init)
  if (failed(updateLoopHeader(funcOp, loopBBs, loopCondition)))
    return signalPassFailure();
}
