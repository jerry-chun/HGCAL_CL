//**************************************************
// \file HGCALTB.cc (batch-only, no visualization)
// \brief: main() of HGCALTB, modified to remove all visual/UI sections
// \author: Adapted for headless mode
//**************************************************

#include "HGCALTBActInitialization.hh"
#include "HGCALTBDetConstruction.hh"

#ifdef G4MULTITHREADED
#  include "G4MTRunManager.hh"
#  include "G4Threading.hh"
#else
#  include "G4RunManager.hh"
#endif

#include "G4PhysListFactory.hh"
#include "G4UImanager.hh"
#include "G4UIcommand.hh"

// CLI string outputs
namespace CLIOutputs {
    void PrintHelp() {
        G4cout << "Usage: HGCALTB -m MACRO [OPTION...]\n"
               << "Options:\n"
               << "  -m MACRO        path to macro file to run (required)\n"
               << "  -t THREADS      number of threads (MT only)\n"
               << "  -p PHYSICSLIST  physics list (default: FTFP_BERT)\n"
               << "  -f FILENAME     custom output filename\n"
               << G4endl;
    }
    void PrintError() {
        G4cerr << "Wrong usage, see 'HGCALTB -h' for help" << G4endl;
    }
}

namespace PrintPLFactoryUsageError {
    void PLFactoryUsageError() {
        G4cerr << "Wrong physics list name. Use a valid reference list." << G4endl;
    }
}

int main(int argc, char** argv) {
    G4String macro;
    G4String custom_pl   = "FTFP_BERT";
    G4String custom_fname = "";
#ifdef G4MULTITHREADED
    G4int nThreads = G4Threading::G4GetNumberOfCores();
#endif

    // Parse CLI
    for (G4int i = 1; i < argc; i += 2) {
        G4String opt = argv[i];
        if      (opt == "-m") macro = argv[++i];
        else if (opt == "-p") custom_pl = argv[++i];
        else if (opt == "-f") custom_fname = argv[++i];
#ifdef G4MULTITHREADED
        else if (opt == "-t") nThreads = G4UIcommand::ConvertToInt(argv[++i]);
#endif
        else if (opt == "-h") { CLIOutputs::PrintHelp(); return 0; }
        else { CLIOutputs::PrintError(); return 1; }
    }

    // Require macro for batch-only
    if (macro.empty()) {
        CLIOutputs::PrintHelp();
        return 1;
    }

    // Run manager
#ifdef G4MULTITHREADED
    auto runManager = new G4MTRunManager;
    if (nThreads > 0) runManager->SetNumberOfThreads(nThreads);
#else
    auto runManager = new G4RunManager;
#endif

    // Physics list
    auto physListFactory = new G4PhysListFactory();
    if (!physListFactory->IsReferencePhysList(custom_pl)) {
        PrintPLFactoryUsageError::PLFactoryUsageError();
        return 1;
    }
    auto physicsList = physListFactory->GetReferencePhysList(custom_pl);
    runManager->SetUserInitialization(physicsList);

    // Detector and actions
    runManager->SetUserInitialization(new HGCALTBDetConstruction());
    runManager->SetUserInitialization(new HGCALTBActInitialization(custom_fname));

    // Execute macro in batch mode (no visualization)
    auto UImanager = G4UImanager::GetUIpointer();
    G4String cmd = "/control/execute ";
    UImanager->ApplyCommand(cmd + macro);

    // Cleanup
    delete runManager;
    return 0;
}
