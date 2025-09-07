#ifndef HGCALTBCHESD_h
#define HGCALTBCHESD_h 1

#include "G4VSensitiveDetector.hh"
#include "HGCALTBCHEHit.hh"
#include <unordered_map>

class G4Step;
class G4HCofThisEvent;

class HGCALTBCHESD : public G4VSensitiveDetector {
public:
    HGCALTBCHESD(const G4String& name);
    virtual ~HGCALTBCHESD() = default;

    virtual void Initialize(G4HCofThisEvent* HCE) override;
    virtual G4bool ProcessHits(G4Step* aStep, G4TouchableHistory* ROhist) override;

    static const G4String fCHEHitsCollectionName;

private:
    HGCALTBCHEHitsCollection* fHitsCollection;

    G4int FindWaferID(G4int cpno) const;

    const std::unordered_map<G4int, G4int> CHEWaferMap = {
        {0, 0}, {10002, 1}, {100101, 2}, {101, 3},
        {10101, 4}, {110101, 5}, {2, 6},
        {33, 7}, {34, 8}, {35, 9}, {36, 10}, {37, 11}, {38, 12}, {39, 13}
    };
};

#endif
