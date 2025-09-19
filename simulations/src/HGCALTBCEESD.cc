#include "HGCALTBCEESD.hh"
#include "HGCALTBConstants.hh"
#include "HGCALTBCEEHit.hh"

#include "TrackPrimaryMap.hh"
#include "G4HCofThisEvent.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4ios.hh"
#include <algorithm> 

#include <unordered_map>
#include <vector>
#include <limits>

extern std::vector<double> gLayerZ;

static inline int ZtoLayer(double z_cm)
{
  auto it = std::lower_bound(gLayerZ.begin(), gLayerZ.end(), z_cm);
  if (it == gLayerZ.end()) return gLayerZ.size();       
  return int(it - gLayerZ.begin()) + 1;                
}




const G4String HGCALTBCEESD::fCEEHitsCollectionName = "CEEHitsCollectionName";

HGCALTBCEESD::HGCALTBCEESD(const G4String& name)
  : G4VSensitiveDetector(name), fHitsCollection(nullptr)
{
  collectionName.insert(fCEEHitsCollectionName);
}

HGCALTBCEESD::~HGCALTBCEESD() {}

void HGCALTBCEESD::Initialize(G4HCofThisEvent* hce)
{
  fHitsCollection = new HGCALTBCEEHitsCollection(SensitiveDetectorName, collectionName[0]);
  auto hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  hce->AddHitsCollection(hcID, fHitsCollection);
}

G4bool HGCALTBCEESD::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
    auto edep = aStep->GetTotalEnergyDeposit();
    if (aStep->GetTrack()->GetGlobalTime() > HGCALTBConstants::TimeCut || edep <= 0.) return false;

    auto pre = aStep->GetPreStepPoint();
    auto touchable = pre->GetTouchableHandle();
    auto pos = pre->GetPosition();

    auto track = aStep->GetTrack();
    G4int trackID  = track->GetTrackID();
    G4int showerID = gTrackToPrimaryMap[trackID];

    double z_cm = pre->GetPosition().z() / CLHEP::cm;
    G4int layer = ZtoLayer(z_cm); 

    auto hit = new HGCALTBCEEHit();
    hit->SetEdep(edep);
    hit->SetPosition(pos.x()/CLHEP::cm, pos.y()/CLHEP::cm, pos.z()/CLHEP::cm);
    hit->SetTrackID(trackID);
    hit->SetShowerID(showerID);
    hit->SetLayer(layer);              
    fHitsCollection->insert(hit);
    return true;
}

void HGCALTBCEESD::EndOfEvent(G4HCofThisEvent*)
{

  const int nHits = fHitsCollection->entries();

  std::unordered_map<int, std::vector<int>> byLayer;
  byLayer.reserve(nHits);

  for (int i = 0; i < nHits; ++i) {
    auto* hit = (*fHitsCollection)[i];
    int L = hit->GetLayer();
    byLayer[L].push_back(i);
  }

  for (auto& kv : byLayer) {
    const int L = kv.first;

    if ((L % 2) != 0) continue;
    if (L <= 1) continue;              
    auto& idxs = kv.second;
    if (idxs.size() < 2) continue;     

    double zmin =  std::numeric_limits<double>::infinity();
    double zmax = -std::numeric_limits<double>::infinity();

    for (int i : idxs) {
      auto* h = (*fHitsCollection)[i];
      double z = h->GetZ();   
      if (z < zmin) zmin = z;
      if (z > zmax) zmax = z;
    }

    if (!(zmax > zmin)) continue;

    const double zmid = 0.5 * (zmin + zmax);

    for (int i : idxs) {
      auto* h = (*fHitsCollection)[i];
      double z = h->GetZ();  
      if (z < zmid) {
        h->SetLayer(L - 1);
      }
    }
  }

  G4cout << "[CEE SD] EndOfEvent called. Total hits: "
         << fHitsCollection->entries() << G4endl;
}
