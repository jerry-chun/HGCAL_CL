#include "HGCALTBRunAction.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4AnalysisManager.hh"

HGCALTBRunAction::HGCALTBRunAction(HGCALTBEventAction* eventAction, G4String filename)
  : G4UserRunAction(), fEventAction(eventAction), fFileName(filename)
{
    G4RunManager::GetRunManager()->SetPrintProgress(1);
    auto analysisManager = G4AnalysisManager::Instance();
    analysisManager->SetVerboseLevel(1);
    analysisManager->SetNtupleMerging(1);

    analysisManager->CreateNtuple("HGCALTBout", "HGCALTBoutput");
    analysisManager->CreateNtupleDColumn("edep");            // 0
    analysisManager->CreateNtupleDColumn("CEETot");          // 1
    analysisManager->CreateNtupleDColumn("CHETot");          // 2
    analysisManager->CreateNtupleDColumn("AHCALTot");        // 3
    analysisManager->CreateNtupleDColumn("HGCALTot");        // 4
    analysisManager->CreateNtupleIColumn("IntLayer");        // 5
    analysisManager->CreateNtupleIColumn("PDGID");           // 6 
    analysisManager->CreateNtupleDColumn("PrimaryEnergy");   // 7 (sum of primaries in MeV)

    analysisManager->CreateNtupleIColumn("CEEIntLayer");     // 8
    analysisManager->CreateNtupleIColumn("CHEIntLayer");     // 9

    analysisManager->CreateNtupleDColumn("CEESignals", fEventAction->GetCEESignals());   // 10
    analysisManager->CreateNtupleDColumn("CHESignals", fEventAction->GetCHESignals());   // 11
    analysisManager->CreateNtupleDColumn("AHCALSignals", fEventAction->GetAHCALSignals());// 12

    analysisManager->CreateNtupleDColumn("hit_x",        fEventAction->hits_x);          // 13
    analysisManager->CreateNtupleDColumn("hit_y",        fEventAction->hits_y);          // 14
    analysisManager->CreateNtupleDColumn("hit_z",        fEventAction->hits_z);          // 15
    analysisManager->CreateNtupleDColumn("hit_Edep",     fEventAction->hits_Edep);       // 16
    analysisManager->CreateNtupleIColumn("hit_trackid",  fEventAction->hit_trackid);     // 17
    analysisManager->CreateNtupleIColumn("hit_showerid", fEventAction->hit_showerid);    // 18
    analysisManager->CreateNtupleIColumn("hit_detector", fEventAction->hit_detector);    // 19
    analysisManager->CreateNtupleIColumn("hit_layer",    fEventAction->hit_layer);       // 20  <-- NEW

    analysisManager->CreateNtupleIColumn("CEEHitCount");                                    // 21
    analysisManager->CreateNtupleDColumn("PrimaryEnergies", fEventAction->GetPrimaryEnergies()); // 22
    analysisManager->CreateNtupleIColumn("PrimaryPDGIDs",   fEventAction->GetPrimaryPDGIDs());   // 23
    analysisManager->FinishNtuple();

}

HGCALTBRunAction::~HGCALTBRunAction()
{
  delete G4AnalysisManager::Instance();
}

void HGCALTBRunAction::BeginOfRunAction(const G4Run* Run)
{
  auto analysisManager = G4AnalysisManager::Instance();
  G4String outputfile = fFileName.empty() ? "HGCALTBout_Run" + std::to_string(Run->GetRunID()) + ".root" : fFileName;
  analysisManager->OpenFile(outputfile);
}

void HGCALTBRunAction::EndOfRunAction(const G4Run*)
{
  auto analysisManager = G4AnalysisManager::Instance();
  analysisManager->Write();
  analysisManager->CloseFile();
}
