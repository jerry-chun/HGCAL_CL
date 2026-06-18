#include "HGCALTBEventAction.hh"
#include "HGCALTBCEESD.hh"
#include "HGCALTBCEEHit.hh"
#include "HGCALTBCHESD.hh"
#include "HGCALTBCHEHit.hh"
#include "HGCALTBAHCALSD.hh"
#include "HGCALTBAHHit.hh"

#include "TrackPrimaryMap.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4SDManager.hh"
#include "G4AnalysisManager.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include <numeric>
#include <algorithm>


HGCALTBEventAction::HGCALTBEventAction(HGCALTBPrimaryGenAction* PGA)
  : G4UserEventAction(), edep(0.), fIntLayer(0), fPrimaryGenAction(PGA), fCEEHCID(-1), fCHEHCID(-1)
{
  fCEELayerSignals = std::vector<G4double>(HGCALTBConstants::CEELayers, 0.);
  fCHELayerSignals = std::vector<G4double>(HGCALTBConstants::CHELayers, 0.);
  fAHCALLayerSignals = std::vector<G4double>(HGCALTBConstants::AHCALLayers, 0.);
}

HGCALTBEventAction::HGCALTBEventAction()
  : G4UserEventAction(), edep(0.), fIntLayer(0), fCEEHCID(-1), fCHEHCID(-1)
{
  fCEELayerSignals = std::vector<G4double>(HGCALTBConstants::CEELayers, 0.);
  fCHELayerSignals = std::vector<G4double>(HGCALTBConstants::CHELayers, 0.);
  fAHCALLayerSignals = std::vector<G4double>(HGCALTBConstants::AHCALLayers, 0.);
}

HGCALTBEventAction::~HGCALTBEventAction() {}

void HGCALTBEventAction::BeginOfEventAction(const G4Event*)
{
  gTrackToPrimaryMap.clear();
  currentShowerIndex = 0;
  edep = 0.;
  cee_hit_count = 0;
  fIntLayer = 0;
  for (auto& v : fCEELayerSignals) v = 0.;
  for (auto& v : fCHELayerSignals) v = 0.;
  for (auto& v : fAHCALLayerSignals) v = 0.;
  hits_x.clear(); hits_y.clear(); hits_z.clear(); hits_Edep.clear();
  hit_trackid.clear(); hit_showerid.clear(); 
  hit_detector.clear();
  hit_layer.clear();
  hit_purity.clear();
}

HGCALTBCEEHitsCollection* HGCALTBEventAction::GetCEEHitsCollection(G4int hcID, const G4Event* event) const
{
  auto hitsCollection = static_cast<HGCALTBCEEHitsCollection*>(event->GetHCofThisEvent()->GetHC(hcID));
  if (!hitsCollection) {
    G4Exception("HGCALTBEventAction::GetCEEHitsCollection", "NoCollection", FatalException, "Cannot access CEE hits");
  }
  return hitsCollection;
}

HGCALTBCHEHitsCollection* HGCALTBEventAction::GetCHEHitsCollection(G4int hcID, const G4Event* event) const {
  auto hitsCollection = static_cast<HGCALTBCHEHitsCollection*>(event->GetHCofThisEvent()->GetHC(hcID));
  if (!hitsCollection) {
    G4Exception("HGCALTBEventAction::GetCHEHitsCollection", "NoCollection", FatalException,
                "Cannot access CHE hits");
  }
  return hitsCollection;
}

HGCALTBAHCALHitsCollection* HGCALTBEventAction::GetAHCALHitsCollection(G4int hcID, const G4Event* event) const
{
  auto hitsCollection = static_cast<HGCALTBAHCALHitsCollection*>(event->GetHCofThisEvent()->GetHC(hcID));
  if (!hitsCollection) {
    G4Exception("HGCALTBEventAction::GetAHCALHitsCollection", "NoCollection", FatalException,
                "Cannot access AHCAL hits");
  }
  return hitsCollection;
}


void HGCALTBEventAction::EndOfEventAction(const G4Event* event)
{
  auto analysisManager = G4AnalysisManager::Instance();

  if (fCEEHCID   == -1) fCEEHCID   = G4SDManager::GetSDMpointer()->GetCollectionID(HGCALTBCEESD::fCEEHitsCollectionName);
  if (fCHEHCID   == -1) fCHEHCID   = G4SDManager::GetSDMpointer()->GetCollectionID(HGCALTBCHESD::fCHEHitsCollectionName);
  if (fAHCALHCID == -1) fAHCALHCID = G4SDManager::GetSDMpointer()->GetCollectionID(HGCALTBAHCALSD::fAHCALHitsCollectionName);


  auto CEEHC = GetCEEHitsCollection(fCEEHCID, event);
  for (G4int i = 0; i < CEEHC->GetSize(); ++i) {
    auto hit = (*CEEHC)[i];
    G4double edep_MeV = hit->GetEdep() / CLHEP::MeV;
    if (edep_MeV > 0.001) {
      hits_x.push_back(hit->GetX());
      hits_y.push_back(hit->GetY());
      hits_z.push_back(hit->GetZ());
      hits_Edep.push_back(edep_MeV);
      hit_trackid.push_back(hit->GetTrackID());
      hit_showerid.push_back(hit->GetShowerID());
      hit_detector.push_back(HGCALDetID::kCEE);   
      hit_layer.push_back(hit->GetLayer());
      hit_purity.push_back(hit->GetPurity());
    }
  }
  cee_hit_count = std::count(hit_detector.begin(), hit_detector.end(), HGCALDetID::kCEE);
  auto CHEHC = GetCHEHitsCollection(fCHEHCID, event);
  if (CHEHC) {
    for (G4int i = 0; i < CHEHC->GetSize(); ++i) {
      auto hit = (*CHEHC)[i];
      G4double edep_MeV = hit->GetEdep() / CLHEP::MeV;
      if (edep_MeV > 0.001) {
        hits_x.push_back(hit->GetX());
        hits_y.push_back(hit->GetY());
        hits_z.push_back(hit->GetZ());
        hits_Edep.push_back(edep_MeV);
        hit_trackid.push_back(hit->GetTrackID());
        hit_showerid.push_back(hit->GetShowerID());
        hit_detector.push_back(HGCALDetID::kCHE);
        hit_layer.push_back(hit->GetLayer());
        hit_purity.push_back(hit->GetPurity());
      }
    }
  }
  std::fill(fAHCALLayerSignals.begin(), fAHCALLayerSignals.end(), 0.0);

  auto AHCALHC = GetAHCALHitsCollection(fAHCALHCID, event);
  if (AHCALHC) {
    // One HGCALTBAHHit per AHCAL layer
    for (G4int il = 0; il < AHCALHC->GetSize(); ++il) {
      const auto* layerHit = (*AHCALHC)[il];
      auto layerTiles = layerHit->GetAHSignals();   
      // sum tiles
      double layerSum = 0.0;
      for (double e : layerTiles) layerSum += e;
      fAHCALLayerSignals[il] = layerSum;            
    }
  }

  double CEETot = 0.0, CHETot = 0.0;
  for (size_t i = 0; i < hits_Edep.size(); ++i) {
    if (hit_detector[i] == HGCALDetID::kCEE) CEETot += hits_Edep[i];
    else if (hit_detector[i] == HGCALDetID::kCHE) CHETot += hits_Edep[i];
  }
  auto AHCALTot = 0;
  // AHCALTot stays from its own SD (or 0 if not attached)
  double HGCALTot = CEETot + CHETot + AHCALTot;
    
  auto& pe = fPrimaryGenAction->GetPrimaryEnergies();
  G4double sumPrimaryMeV = std::accumulate(pe.begin(), pe.end(), 0.0);

  analysisManager->FillNtupleDColumn(0, edep);
  analysisManager->FillNtupleDColumn(1, CEETot);
  analysisManager->FillNtupleDColumn(2, CHETot);
  analysisManager->FillNtupleDColumn(3, AHCALTot);
  analysisManager->FillNtupleDColumn(4, HGCALTot);
  analysisManager->FillNtupleIColumn(5, fIntLayer);
  analysisManager->FillNtupleIColumn(6, fPrimaryGenAction->GetParticleGun()->GetParticleDefinition()->GetPDGEncoding());
  analysisManager->FillNtupleDColumn(7, sumPrimaryMeV);  
  analysisManager->FillNtupleIColumn(8, 0);
  analysisManager->FillNtupleIColumn(9, 0);
  //analysisManager->FillNtupleIColumn(23, cee_hit_count);
  analysisManager->AddNtupleRow();

  auto che_hits = std::count(hit_detector.begin(), hit_detector.end(), HGCALDetID::kCHE);
  G4cout << "[EventAction] CHE hits stored: " << che_hits << G4endl;
}
