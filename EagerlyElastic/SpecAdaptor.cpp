#include "SpecAdaptor.h"
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
#define GEN_PASS_DEF_specADAPTOR
#include "experimental/Transforms/Passes.h.inc"

} // namespace EE
} // namespace experimental
} // namespace dynamatic

struct SpecAdaptorPass
    : public dynamatic::experimental::EE::impl::SpecAdaptorBase<
          SpecAdaptorPass> {
  using SpecAdaptorBase<SpecAdaptorPass>::SpecAdaptorBase;
  void runDynamaticPass() override;
};

void SpecAdaptorPass::runDynamaticPass() {
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

  OpBuilder builder(funcOp.getContext());

  for (unsigned bb : loopBBs) {
    if (hasBranch(funcOp, bb)) {
      if (isExitingBBWithBranch(funcOp, bb, loopBBs)) {
        exitingBBs.push_back(bb);
      } else {
        introduceGSAMux(funcOp, bb);

        for (auto branch : funcOp.getOps<ConditionalBranchOp>()) {
          auto brBB = getLogicBB(branch);
          if (!brBB || *brBB != bb)
            continue;

          branch->setAttr("spec_adaptor_inner_loop", builder.getBoolAttr(true));
        }
      }
    }
  }

  for (auto cmerge : funcOp.getOps<ControlMergeOp>()) {
    auto cmergeBB = getLogicBB(cmerge);
    if (!cmergeBB || *cmergeBB == loopBBs[0] ||
        llvm::find(loopBBs, *cmergeBB) == loopBBs.end())
      continue;

    cmerge->setAttr("spec_adaptor_inner_loop", builder.getBoolAttr(true));
  }

  for (auto mux : funcOp.getOps<MuxOp>()) {
    auto muxBB = getLogicBB(mux);
    if (!muxBB || *muxBB == loopBBs[0] ||
        llvm::find(loopBBs, *muxBB) == loopBBs.end())
      continue;

    mux->setAttr("spec_adaptor_inner_loop", builder.getBoolAttr(true));
  }

  DenseMap<unsigned, unsigned> bbMap = unifyBBs(loopBBs, funcOp);
  if (bbMapping != "") {
    // Convert bbMap to a json file
    std::ofstream csvFile(bbMapping);
    csvFile << "before,after\n";
    for (const auto &entry : bbMap) {
      csvFile << entry.first << "," << entry.second << "\n";
    }
    csvFile.close();
  }

  recalculateMCBlocks(funcOp);
}
