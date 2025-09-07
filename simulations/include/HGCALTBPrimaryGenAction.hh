// --------------------------------------------------
#ifndef HGCALTBPrimaryGenAction_h
#define HGCALTBPrimaryGenAction_h 1

// Geant4 core
#include "G4Types.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4GenericMessenger.hh"

// STL
#include <vector>
#include <string>
#include "G4String.hh"

// Forward declarations
#include "G4ParticleGun.hh"
class G4Event;

class HGCALTBPrimaryGenAction : public G4VUserPrimaryGeneratorAction {
public:
    HGCALTBPrimaryGenAction();
    ~HGCALTBPrimaryGenAction() override;

    void GeneratePrimaries(G4Event* event) override;
    const G4ParticleGun* GetParticleGun() const;
    
    std::vector<G4double>& GetPrimaryEnergies() { return fPrimaryEnergiesMeV; } // MeV
    std::vector<G4int>&    GetPrimaryPDGIDs()   { return fPrimaryPDGIDs; }

private:
    G4ParticleGun*      fParticleGun;
    G4GenericMessenger  fMessenger;

    // configurable ranges
    G4int               fNumParticlesMin;
    G4int               fNumParticlesMax;
    G4double            fEnergyMin;
    G4double            fEnergyMax;
    G4double            fThetaMin;
    G4double            fThetaMax;
    G4double            fZMin;
    G4double            fZMax;

    // stored as string, parsed into vector on each event
    G4String            fParticleTypesStr;

    // helper to split fParticleTypesStr
    std::vector<G4String> ParseParticleTypes() const;
    
    std::vector<G4double> fPrimaryEnergiesMeV; // store in MeV for ntuple
    std::vector<G4int>    fPrimaryPDGIDs;
};

inline const G4ParticleGun* HGCALTBPrimaryGenAction::GetParticleGun() const {
    return fParticleGun;
}

#endif  // HGCALTBPrimaryGenAction_h



