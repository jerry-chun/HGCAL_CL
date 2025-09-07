#include "HGCALTBCHESD.hh"
#include "G4Step.hh"
#include "G4HCofThisEvent.hh"
#include "G4SDManager.hh"
#include "G4TouchableHistory.hh"
#include "G4ThreeVector.hh"
#include "G4ios.hh"
#include "G4SystemOfUnits.hh"
#include "TrackPrimaryMap.hh"
#include <algorithm> 

extern std::vector<double> gLayerZ;

static inline int ZtoLayer(double z_cm)
{
  auto it = std::lower_bound(gLayerZ.begin(), gLayerZ.end(), z_cm);
  if (it == gLayerZ.end()) return gLayerZ.size();        
  return int(it - gLayerZ.begin()) + 1;                  
}





const G4String HGCALTBCHESD::fCHEHitsCollectionName = "CHEHitsCollectionName";

HGCALTBCHESD::HGCALTBCHESD(const G4String& name)
    : G4VSensitiveDetector(name), fHitsCollection(nullptr) {
    collectionName.insert(fCHEHitsCollectionName);
}

void HGCALTBCHESD::Initialize(G4HCofThisEvent* HCE) {
    fHitsCollection = new HGCALTBCHEHitsCollection(SensitiveDetectorName, fCHEHitsCollectionName);
    G4int HCID = G4SDManager::GetSDMpointer()->GetCollectionID(fHitsCollection);
    HCE->AddHitsCollection(HCID, fHitsCollection);
}

G4bool HGCALTBCHESD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
    auto edep = aStep->GetTotalEnergyDeposit();
    if (edep == 0.) return false;

    auto pre = aStep->GetPreStepPoint();
    auto touchable = pre->GetTouchableHandle();
    auto pos = pre->GetPosition();

    auto track = aStep->GetTrack();
    G4int trackID  = track->GetTrackID();
    G4int showerID = gTrackToPrimaryMap[trackID];

    double z_cm = pre->GetPosition().z() / CLHEP::cm;
    G4int layer = ZtoLayer(z_cm);       

    auto hit = new HGCALTBCHEHit();
    hit->SetPosition(pos.x()/CLHEP::cm, pos.y()/CLHEP::cm, pos.z()/CLHEP::cm);
    hit->SetEdep(edep);
    hit->SetTrackID(trackID);
    hit->SetShowerID(showerID);
    hit->SetLayer(layer);               
    
    // *This does not work, trying to save if scintillator. Not sure needed.*
    //bool isScint = false;
    //if (auto* lv0 = touchable->GetVolume(0) ? touchable->GetVolume(0)->GetLogicalVolume() : nullptr) {
    //  if (auto* mat0 = lv0->GetMaterial()) {
    //    const G4String& mn = mat0->GetName();
    //    isScint = (mn.find("SC_VINYLTOLUENE") != G4String::npos) ||
    //              (mn.find("POLYSTYRENE")     != G4String::npos) ||
    //              (mn.find("Scint")           != G4String::npos);
    //  }
    //}
    //hit->SetIsScint(isScint);
    
    fHitsCollection->insert(hit);
    return true;

}

G4int HGCALTBCHESD::FindWaferID(G4int cpno) const {
    auto key = CHEWaferMap.find(cpno);
    if (key != CHEWaferMap.end()) {
        return key->second;
    } else {
        G4cout << "[CHE WARNING] Wafer cpno " << cpno << " not in map — defaulting to 0" << G4endl;
        return 0;
    }
}
