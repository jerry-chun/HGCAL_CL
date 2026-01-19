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
#include <cmath>   // std::lround

namespace {
  struct LayerRange {
    double z_min;
    double z_max;
  };

  static const LayerRange kLayerRanges[50] = {
    {319.666, 322.166},
    {322.166, 324.665},
    {324.665, 327.164},
    {327.164, 329.664},
    {329.664, 332.165},
    {332.165, 334.665},
    {334.665, 337.165},
    {337.165, 339.666},
    {339.666, 342.166},
    {342.166, 344.666},
    {344.666, 347.165},
    {347.165, 349.665},
    {349.665, 352.165},
    {352.165, 354.666},
    {354.666, 357.166},
    {357.166, 359.666},
    {359.666, 362.166},
    {362.166, 364.665},
    {364.665, 367.166},
    {367.166, 369.666},
    {369.666, 372.166},
    {372.166, 374.666},
    {374.666, 377.165},
    {377.165, 379.665},
    {379.665, 382.165},
    {382.165, 384.665},
    {384.665, 387.165},
    {387.165, 391.416},
    {391.416, 396.916},
    {396.916, 401.915},
    {401.915, 406.916},
    {406.916, 411.916},
    {411.916, 416.916},
    {416.916, 421.916},
    {421.916, 426.916},
    {426.916, 431.916},
    {431.916, 436.916},
    {436.916, 441.916},
    {441.916, 446.916},
    {446.916, 451.916},
    {451.916, 456.915},
    {456.915, 461.915},
    {461.915, 466.916},
    {466.916, 471.916},
    {471.916, 476.916},
    {476.916, 481.914},
    {481.914, 486.914},
    {486.914, 491.916},
    {491.916, 496.918},
    {496.918, 501.922}
  };

  inline int ZtoLayer(double z_cm)
  {
    constexpr double eps = 1e-4; 
    for (int i = 0; i < 50; ++i) {
      if (z_cm >= kLayerRanges[i].z_min - eps &&
          z_cm <= kLayerRanges[i].z_max + eps) {
        return i + 1;  // layers numbered from 1
      }
    }
    return 0; // no valid layer
  }

} // anonymous namespace

const G4String HGCALTBCHESD::fCHEHitsCollectionName = "CHEHitsCollectionName";

HGCALTBCHESD::HGCALTBCHESD(const G4String& name)
    : G4VSensitiveDetector(name), fHitsCollection(nullptr) {
    collectionName.insert(fCHEHitsCollectionName);
}

void HGCALTBCHESD::Initialize(G4HCofThisEvent* HCE) {
    fHitsCollection =
      new HGCALTBCHEHitsCollection(SensitiveDetectorName, fCHEHitsCollectionName);

    G4int HCID = G4SDManager::GetSDMpointer()->GetCollectionID(fHitsCollection);
    HCE->AddHitsCollection(HCID, fHitsCollection);

    // clear per-event map
    fCellMap.clear();
}

G4bool HGCALTBCHESD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
    auto edep = aStep->GetTotalEnergyDeposit();
    if (edep == 0.) return false;

    auto pre   = aStep->GetPreStepPoint();
    auto track = aStep->GetTrack();

    G4int trackID  = track->GetTrackID();
    G4int showerID = gTrackToPrimaryMap[trackID];

    double z_step_cm = pre->GetPosition().z() / CLHEP::cm;
    G4int layer = ZtoLayer(z_step_cm);
    if (layer <= 0) {
        return false;
    }

    const auto touchable = pre->GetTouchableHandle();
    G4ThreeVector globalCenter = touchable->GetTranslation();

    double cx_cm = globalCenter.x() / CLHEP::cm;
    double cy_cm = globalCenter.y() / CLHEP::cm;
    double cz_cm = globalCenter.z() / CLHEP::cm;

    int ix = static_cast<int>(std::lround(cx_cm));
    int iy = static_cast<int>(std::lround(cy_cm));

    CHECellKey key{layer, ix, iy};
    auto& acc = fCellMap[key];  // creates if not present

    if (!acc.hasCenter) {
        acc.cx = cx_cm;
        acc.cy = cy_cm;
        acc.cz = cz_cm;
        acc.hasCenter = true;
    }
    if (acc.firstTrackID < 0) {
        acc.firstTrackID = trackID;
    }

    acc.edep += edep;

    bool found = false;
    for (auto& sh : acc.showerContribs) {
        if (sh.first == showerID) {
            sh.second += edep;
            found = true;
            break;
        }
    }
    if (!found) {
        acc.showerContribs.emplace_back(showerID, edep);
    }

    // don't create a hit yet; we do that in EndOfEvent
    return true;
}

void HGCALTBCHESD::EndOfEvent(G4HCofThisEvent*) {
    // turn each accumulated cell into one hit
    for (const auto& kv : fCellMap) {
        const CHECellKey&   key = kv.first;
        const CHECellAccum& acc = kv.second;

        if (acc.edep <= 0.0 || !acc.hasCenter) continue;

        auto hit = new HGCALTBCHEHit();

        hit->SetPosition(acc.cx, acc.cy, acc.cz);
        hit->SetEdep(acc.edep);
        hit->SetTrackID(acc.firstTrackID);
        hit->SetLayer(key.layer);

        int    dominantShowerID = -1;
        double maxE             = 0.0;
        for (const auto& sh : acc.showerContribs) {
            if (sh.second > maxE) {
                maxE = sh.second;
                dominantShowerID = sh.first;
            }
        } 
        
        double purity = 0.0;
        if (acc.edep > 0.0 && maxE > 0.0) {
            purity = maxE / acc.edep;
            if (purity > 1.0) purity = 1.0;  // guard against rounding
        }
       
        hit->SetShowerID(dominantShowerID);
        hit->SetPurity(purity);

        fHitsCollection->insert(hit);
    }

    G4cout << "[CHE SD] EndOfEvent: merged to "
           << fHitsCollection->entries()
           << " hits (one per cell per layer, at true cell centers)"
           << G4endl;

    fCellMap.clear();
}

G4int HGCALTBCHESD::FindWaferID(G4int cpno) const {
    auto key = CHEWaferMap.find(cpno);
    if (key != CHEWaferMap.end()) {
        return key->second;
    } else {
        G4cout << "[CHE WARNING] Wafer cpno " << cpno
               << " not in map — defaulting to 0" << G4endl;
        return 0;
    }
}
