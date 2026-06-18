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
    analysisManager->CreateNtupleDColumn("edep");            
    analysisManager->CreateNtupleDColumn("CEETot");          
    analysisManager->CreateNtupleDColumn("CHETot");          
    analysisManager->CreateNtupleDColumn("AHCALTot");        
    analysisManager->CreateNtupleDColumn("HGCALTot");        
    analysisManager->CreateNtupleIColumn("IntLayer");        
    analysisManager->CreateNtupleIColumn("PDGID");           
    analysisManager->CreateNtupleDColumn("PrimaryEnergy"); 

    analysisManager->CreateNtupleIColumn("CEEIntLayer");     
    analysisManager->CreateNtupleIColumn("CHEIntLayer");    
    analysisManager->CreateNtupleDColumn("CEESignals", fEventAction->GetCEESignals());   
    analysisManager->CreateNtupleDColumn("CHESignals", fEventAction->GetCHESignals());   
    analysisManager->CreateNtupleDColumn("AHCALSignals", fEventAction->GetAHCALSignals());

    analysisManager->CreateNtupleDColumn("hit_x",        fEventAction->hits_x);          
    analysisManager->CreateNtupleDColumn("hit_y",        fEventAction->hits_y);  
    analysisManager->CreateNtupleDColumn("hit_z",        fEventAction->hits_z);          
    analysisManager->CreateNtupleDColumn("hit_Edep",     fEventAction->hits_Edep);       
    analysisManager->CreateNtupleIColumn("hit_trackid",  fEventAction->hit_trackid);     
    analysisManager->CreateNtupleIColumn("hit_showerid", fEventAction->hit_showerid);    
    analysisManager->CreateNtupleIColumn("hit_detector", fEventAction->hit_detector);    
    analysisManager->CreateNtupleIColumn("hit_layer",    fEventAction->hit_layer);       
    analysisManager->CreateNtupleDColumn("hit_purity",   fEventAction->hit_purity);      


    analysisManager->CreateNtupleIColumn("CEEHitCount");                                    
    analysisManager->CreateNtupleDColumn("PrimaryEnergies", fEventAction->GetPrimaryEnergies()); 
    analysisManager->CreateNtupleIColumn("PrimaryPDGIDs",   fEventAction->GetPrimaryPDGIDs());   
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
